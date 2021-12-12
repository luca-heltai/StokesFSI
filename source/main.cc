#include "StokesFSI.h"



template <int dim>
StokesFSI<dim>::StokesFSI(const unsigned int degree)
  : degree(degree)
  , triangulation(Triangulation<dim>::maximum_smoothing)
  , fe(FE_Q<dim>(degree + 1), dim, FE_Q<dim>(degree), 1)
  , dh(triangulation)
  , time(0., 10., time_step)
{}

template <int dim>
void
StokesFSI<dim>::make_grid()
{
  const Point<dim> top_right = (dim == 2 ? Point<dim>(length, height) : // 2d
                                  Point<dim>(length, width, height));   // 3d

  GridGenerator::hyper_rectangle(triangulation, Point<dim>(), top_right);
  triangulation.refine_global(refinements);

  auto near = [](const auto &a, const auto &b) {
    return std::abs(a - b) < 1e-10;
  };

  double cells_per_side = pow(triangulation.n_active_cells(), 1. / dim);

  material_mask.reinit(triangulation.n_active_cells());
  for (const auto &cell : triangulation.active_cell_iterators())
    {
      Point<dim> cell_center = cell->center();

      if (abs(cell_center[0] - (top_right[0] / 2)) <
            (top_right[0] / cells_per_side) &&
          cell_center[dim - 1] <
            solid_liquid_height_proportion * top_right[dim - 1])
        {
          if (dim == 2)
            cell->set_material_id(1);
          else
            {
              if (abs(cell_center[1] - (top_right[1] / 2)) <
                  (top_right[1] / cells_per_side))
                cell->set_material_id(1);
              else
                cell->set_material_id(0);
            }
        }
      else
        cell->set_material_id(0);

      material_mask[cell->index()] = cell->material_id();

      for (const auto &face : cell->face_iterators())
        if (face->at_boundary())
          {
            Point<dim> face_center = face->center();

            if (near(face_center[dim - 1], 0))
              face->set_all_boundary_ids(1); // bottom

            else if (near(face_center[0], 0.) || near(face_center[0], length) ||
                     (dim == 3 && (near(face_center[1], 0.) ||
                                   near(face_center[1], width))))
              face->set_all_boundary_ids(2); // side

            else
              {
                face->set_all_boundary_ids(0); // top}
              }
          }
    }

  std::ofstream out("reference_grid_" + Utilities::int_to_string(dim) +
                    "D.vtk");
  GridOut       grid_out;
  grid_out.write_vtk(triangulation, out);
  std::cout << "Grid saved." << std::endl;
}



template <int dim>
void
StokesFSI<dim>::setup_dofs()
{
  dh.distribute_dofs(fe);
  DoFRenumbering::Cuthill_McKee(dh);

  std::vector<unsigned int> block_component(dim + 1, 0);
  block_component[dim] = 1;
  DoFRenumbering::component_wise(dh, block_component);

  {
    constraints.clear();
    FEValuesExtractors::Vector velocities(0);
    DoFTools::make_hanging_node_constraints(dh, constraints);

    DoFTools::make_zero_boundary_constraints(dh,
                                             1,
                                             constraints,
                                             fe.component_mask(velocities));

    VectorTools::compute_no_normal_flux_constraints(dh, 0, {2}, constraints);
  }
  constraints.close();

  const std::vector<types::global_dof_index> dofs_per_block =
    DoFTools::count_dofs_per_fe_block(dh, block_component);
  const unsigned int n_v = dofs_per_block[0];
  const unsigned int n_p = dofs_per_block[1];
  std::cout << "   Number of active cells: " << triangulation.n_active_cells()
            << std::endl
            << "   Number of degrees of freedom: " << dh.n_dofs() << " (" << n_v
            << '+' << n_p << ')' << std::endl;

  {
    BlockDynamicSparsityPattern dsp(dofs_per_block, dofs_per_block);

    Table<2, DoFTools::Coupling> coupling(dim + 1, dim + 1);

    for (unsigned int c = 0; c < dim + 1; ++c)
      for (unsigned int d = 0; d < dim + 1; ++d)
        if (!((c == dim) && (d == dim)))
          coupling[c][d] = DoFTools::always;
        else
          coupling[c][d] = DoFTools::none;

    DoFTools::make_sparsity_pattern(dh, coupling, dsp, constraints, false);
    sparsity_pattern.copy_from(dsp);
  }

  {
    BlockDynamicSparsityPattern preconditioner_dsp(dofs_per_block,
                                                   dofs_per_block);

    Table<2, DoFTools::Coupling> preconditioner_coupling(dim + 1, dim + 1);

    for (unsigned int c = 0; c < dim + 1; ++c)
      for (unsigned int d = 0; d < dim + 1; ++d)
        if (((c == dim) && (d == dim)))
          preconditioner_coupling[c][d] = DoFTools::always;
        else
          preconditioner_coupling[c][d] = DoFTools::none;
    DoFTools::make_sparsity_pattern(
      dh, preconditioner_coupling, preconditioner_dsp, constraints, false);
    preconditioner_sparsity_pattern.copy_from(preconditioner_dsp);
  }

  system_matrix.reinit(sparsity_pattern);
  preconditioner_matrix.reinit(preconditioner_sparsity_pattern);

  solution.reinit(dofs_per_block);
  system_rhs.reinit(dofs_per_block);
  displacement.reinit(dofs_per_block);

  displacement_mapping =
    std::make_unique<MappingQEulerian<dim, BlockVector<double>>>(degree + 1,
                                                                 dh,
                                                                 displacement);
}



template <int dim>
void
StokesFSI<dim>::assemble_system()
{
  system_matrix         = 0;
  system_rhs            = 0;
  preconditioner_matrix = 0;
  QGauss<dim> quadrature_formula(degree + 2);

  FEValues<dim>      fe_values(*displacement_mapping,
                          fe,
                          quadrature_formula,
                          update_values | update_quadrature_points |
                            update_JxW_values | update_gradients);
  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  const unsigned int n_q_points    = quadrature_formula.size();
  FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> local_preconditioner_matrix(dofs_per_cell, dofs_per_cell);

  Vector<double>                       local_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  const RightHandSide<dim>             right_hand_side;
  std::vector<Vector<double>> rhs_values(n_q_points, Vector<double>(dim + 1));

  const FEValuesExtractors::Vector velocities(0);
  const FEValuesExtractors::Scalar pressure(dim);

  std::vector<SymmetricTensor<2, dim>> symgrad_phi_u(dofs_per_cell);
  std::vector<double>                  div_phi_u(dofs_per_cell);
  std::vector<double>                  phi_p(dofs_per_cell);

  for (const auto &cell : dh.active_cell_iterators())
    {
      double lambda = (cell->material_id() == 1 ? time_step * mu : eta);

      fe_values.reinit(cell);
      local_matrix                = 0;
      local_preconditioner_matrix = 0;
      local_rhs                   = 0;
      right_hand_side.vector_value_list(fe_values.get_quadrature_points(),
                                        rhs_values);

      for (unsigned int q = 0; q < n_q_points; ++q)
        {
          for (unsigned int k = 0; k < dofs_per_cell; ++k)
            {
              symgrad_phi_u[k] = fe_values[velocities].symmetric_gradient(k, q);
              div_phi_u[k]     = fe_values[velocities].divergence(k, q);
              phi_p[k]         = fe_values[pressure].value(k, q);
            }

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  local_matrix(i, j) +=
                    (lambda * (symgrad_phi_u[i] * symgrad_phi_u[j]) // (1)
                     - div_phi_u[i] * phi_p[j]                      // (2)
                     - phi_p[i] * div_phi_u[j])                     // (3)
                    * fe_values.JxW(q);                             // * dx
                  local_preconditioner_matrix(i, j) +=
                    (phi_p[i] * phi_p[j]) // (4)
                    * fe_values.JxW(q);   // * dx
                }

              const unsigned int component_i =
                fe.system_to_component_index(i).first;
              local_rhs(i) += (fe_values.shape_value(i, q)   // (phi_u_i(x_q)
                               * rhs_values[q](component_i)) // * f(x_q))
                              * fe_values.JxW(q);            // * dx
            }
        }


      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        for (unsigned int j = i + 1; j < dofs_per_cell; ++j)
          {
            local_matrix(i, j) = local_matrix(j, i);
            local_preconditioner_matrix(i, j) =
              local_preconditioner_matrix(j, i);
          }

      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(
        local_matrix, local_rhs, local_dof_indices, system_matrix, system_rhs);

      constraints.distribute_local_to_global(local_preconditioner_matrix,
                                             local_dof_indices,
                                             preconditioner_matrix);
    }

  std::cout << "   Computing preconditioner..." << std::endl << std::flush;

  A_preconditioner =
    std::make_shared<typename InnerPreconditioner<dim>::type>();
  A_preconditioner->initialize(
    system_matrix.block(0, 0),
    typename InnerPreconditioner<dim>::type::AdditionalData());
}



template <int dim>
void
StokesFSI<dim>::solve()
{
  //   // Let's first try a simple direct solver.
  //   // This worked. There were issues with the numbering.
  //   SparseDirectUMFPACK A_direct;
  //   A_direct.initialize(system_matrix);
  //   A_direct.vmult(solution, system_rhs);
  //   constraints.distribute(solution);

  const InverseMatrix<SparseMatrix<double>,
                      typename InnerPreconditioner<dim>::type>
    A_inverse(system_matrix.block(0, 0), *A_preconditioner);

  Vector<double> tmp(solution.block(0).size());
  {
    Vector<double> schur_rhs(solution.block(1).size());
    A_inverse.vmult(tmp, system_rhs.block(0));
    system_matrix.block(1, 0).vmult(schur_rhs, tmp);
    schur_rhs -= system_rhs.block(1);
    SchurComplement<typename InnerPreconditioner<dim>::type> schur_complement(
      system_matrix, A_inverse);

    SolverControl            solver_control(solution.block(1).size(),
                                 1e-6 * schur_rhs.l2_norm());
    SolverCG<Vector<double>> cg(solver_control);
    SparseILU<double>        preconditioner;
    preconditioner.initialize(preconditioner_matrix.block(1, 1),
                              SparseILU<double>::AdditionalData());
    InverseMatrix<SparseMatrix<double>, SparseILU<double>> m_inverse(
      preconditioner_matrix.block(1, 1), preconditioner);
    cg.solve(schur_complement, solution.block(1), schur_rhs, m_inverse);
    constraints.distribute(solution);
    std::cout << "  " << solver_control.last_step()
              << " outer CG Schur complement iterations for pressure"
              << std::endl;
  }
  {
    system_matrix.block(0, 1).vmult(tmp, solution.block(1));
    tmp *= -1;
    tmp += system_rhs.block(0);
    A_inverse.vmult(solution.block(0), tmp);
    constraints.distribute(solution);
  }
}



template <int dim>
void
StokesFSI<dim>::update_displacement()
{
  displacement.block(0).sadd(1.0, time_step, solution.block(0));
}



template <int dim>
void
StokesFSI<dim>::output_results(const unsigned int step) const
{
  std::vector<std::string> solution_names(dim, "velocity");
  solution_names.emplace_back("pressure");

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation(
      dim, DataComponentInterpretation::component_is_part_of_vector);
  data_component_interpretation.push_back(
    DataComponentInterpretation::component_is_scalar);

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dh);
  data_out.add_data_vector(solution,
                           solution_names,
                           DataOut<dim>::type_dof_data,
                           data_component_interpretation);

  data_out.add_data_vector(material_mask,
                           "material",
                           DataOut<dim>::type_cell_data);

  data_out.build_patches(*displacement_mapping);
  auto name = "solution" + Utilities::int_to_string(dim, 1) + "D--" +
              Utilities::int_to_string(step, 3) + ".vtu";
  std::ofstream output(name);
  data_out.write_vtu(output);


  // Output pvd record.
  static std::vector<std::pair<double, std::string>> times_and_names;
  times_and_names.push_back(std::make_pair(step * time_step, name));

  std::ofstream pvd_output("solution_" + Utilities::int_to_string(dim, 1) +
                           "D.pvd");
  DataOutBase::write_pvd_record(pvd_output, times_and_names);


  /*
  MappingQEulerian<dim> q2_mapping(2, dh_u, old_displacement);
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dh_u);
  data_out.add_data_vector(old_displacement,
                          "Soluzione");
  data_out.build_patches(q2_mapping);

  std::ofstream output(dim == 2 ? "solution-2d.vtk" : "solution-3d.vtk");
  data_out.write_vtk(output);

  std::string s = std::to_string(dim);
  std::ofstream out("grid" + s + ".vtk");
  GridOut       grid_out;
  grid_out.write_vtk(triangulation, out);
  std::cout << "Grid written" << std::endl;
  */
}

template <int dim>
void
StokesFSI<dim>::run()
{
  make_grid();
  setup_dofs();

  VectorTools::project(dh,
                       constraints,
                       QGauss<dim>(degree + 2),
                       InitialValues<dim>(),
                       displacement);

  output_results(0);

  do
    {
      unsigned int step = time.get_step_number() + 1;
      std::cout << "Timestep " << step << std::endl;
      assemble_system();
      solve();
      output_results(step);
      update_displacement();

      time.advance_time();
      std::cout << "   Now at t=" << time.get_current_time()
                << ", dt=" << time.get_previous_step_size() << '.' << std::endl
                << std::endl;
    }
  while (time.is_at_end() == false);

  std::cout << Utilities::int_to_string(dim, 1) + "D run finished." << std::endl
            << std::endl
            << std::endl;
}



int
main()
{
  try
    {
      StokesFSI<2> problem(1);
      problem.run();
      // StokesFSI<3> problems(1);
      // problems.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  return 0;
}
