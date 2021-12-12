#include <deal.II/base/discrete_time.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q_eulerian.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_ilu.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>


using namespace dealii;

/* ------------------------------- */

template <int dim>
struct InnerPreconditioner;

template <>
struct InnerPreconditioner<2>
{
  using type = SparseDirectUMFPACK;
};

template <>
struct InnerPreconditioner<3>
{
  using type = SparseILU<double>;
};

/* ------------------------------- */

template <class MatrixType, class PreconditionerType>
class InverseMatrix : public Subscriptor
{
public:
  InverseMatrix(const MatrixType &m, const PreconditionerType &preconditioner);
  void
  vmult(Vector<double> &dst, const Vector<double> &src) const;

private:
  const SmartPointer<const MatrixType>         matrix;
  const SmartPointer<const PreconditionerType> preconditioner;
};
template <class MatrixType, class PreconditionerType>
InverseMatrix<MatrixType, PreconditionerType>::InverseMatrix(
  const MatrixType &        m,
  const PreconditionerType &preconditioner)
  : matrix(&m)
  , preconditioner(&preconditioner)
{}

template <class MatrixType, class PreconditionerType>
void
InverseMatrix<MatrixType, PreconditionerType>::vmult(
  Vector<double> &      dst,
  const Vector<double> &src) const
{
  SolverControl            solver_control(src.size(), 1e-6 * src.l2_norm());
  SolverCG<Vector<double>> cg(solver_control);
  dst = 0;
  cg.solve(*matrix, dst, src, *preconditioner);
}

/* ------------------------------- */

template <class PreconditionerType>
class SchurComplement : public Subscriptor
{
public:
  SchurComplement(
    const BlockSparseMatrix<double> &system_matrix,
    const InverseMatrix<SparseMatrix<double>, PreconditionerType> &A_inverse);
  void
  vmult(Vector<double> &dst, const Vector<double> &src) const;

private:
  const SmartPointer<const BlockSparseMatrix<double>> system_matrix;
  const SmartPointer<
    const InverseMatrix<SparseMatrix<double>, PreconditionerType>>
                         A_inverse;
  mutable Vector<double> tmp1, tmp2;
};

template <class PreconditionerType>
SchurComplement<PreconditionerType>::SchurComplement(
  const BlockSparseMatrix<double> &                              system_matrix,
  const InverseMatrix<SparseMatrix<double>, PreconditionerType> &A_inverse)
  : system_matrix(&system_matrix)
  , A_inverse(&A_inverse)
  , tmp1(system_matrix.block(0, 0).m())
  , tmp2(system_matrix.block(0, 0).m())
{}
template <class PreconditionerType>
void
SchurComplement<PreconditionerType>::vmult(Vector<double> &      dst,
                                           const Vector<double> &src) const
{
  system_matrix->block(0, 1).vmult(tmp1, src);
  A_inverse->vmult(tmp2, tmp1);
  system_matrix->block(1, 0).vmult(dst, tmp2);
}

/* ------------------------------- */


// PARAMETERS


// Size of the "pool"
const double length = 10.; // x
const double width  = 3.;  // y
const double height = 5.;  // z

const double       max_displacement               = 1;
const double       solid_liquid_height_proportion = 1. / 2.;
const unsigned int refinements                    = 5;

const double time_step = .1;
const double mu        = 1000.;
const double eta       = 1.0;


/* ------------------------------- */

template <int dim>
class InitialValues : public Function<dim>
{
public:
  InitialValues()
    : Function<dim>(dim + 1)
  {}

  virtual double
  value(const Point<dim> &p, const unsigned int component) const override
  {
    const double XI = height * solid_liquid_height_proportion;

    if (component == 0)
      return 0;
    else if (dim == 3 and component == 1)
      return 0;
    else if (component == dim - 1)
      {
        if (p[component] < XI)
          return 0;
        else
          return (1. - (2. * p[0]) / 10.) * max_displacement *
                 (p[dim - 1] - XI) / (height - XI);
      }
    else if (component == dim)
      return 0.;
    return 0;
  }

  virtual void
  vector_value(const Point<dim> &p, Vector<double> &values) const override
  {
    for (unsigned int c = 0; c < this->n_components; ++c)
      values(c) = this->value(p, c);
  }
};

/* ------------------------------- */

template <int dim>
class RightHandSide : public Function<dim>
{
public:
  RightHandSide()
    : Function<dim>(dim + 1)
  {}

  virtual double
  value(const Point<dim> &p, const unsigned int material) const override;
  virtual void
  vector_value(const Point<dim> &p, Vector<double> &value) const override;
};


/* ------------------------------- */

template <int dim>
double
RightHandSide<dim>::value(const Point<dim> & /*p*/,
                          const unsigned int component) const
{
  Assert(component < this->n_components,
         ExcIndexRange(component, 0, this->n_components));
  if (component == 1)
    return -1.;
  return 0;
}

template <int dim>
void
RightHandSide<dim>::vector_value(const Point<dim> &p,
                                 Vector<double> &  values) const
{
  for (unsigned int c = 0; c < this->n_components; ++c)
    values(c) = RightHandSide<dim>::value(p, c);
}

/* ------------------------------- */



template <int dim>
class StokesFSI
{
public:
  StokesFSI(const unsigned int degree);
  void
  run();


private:
  const unsigned int degree;

  void
  make_grid();
  void
  setup_dofs();
  void
  assemble_system();
  void
  solve();
  void
  update_displacement();
  void
  output_results(const unsigned int refinement_cycle) const;



  Triangulation<dim> triangulation;
  FESystem<dim>      fe;

  DoFHandler<dim> dh;

  AffineConstraints<double> constraints;

  BlockSparsityPattern      sparsity_pattern;
  BlockSparseMatrix<double> system_matrix;
  BlockSparsityPattern      preconditioner_sparsity_pattern;
  BlockSparseMatrix<double> preconditioner_matrix;

  BlockVector<double> solution;
  BlockVector<double> system_rhs;

  Vector<double> material_mask;

  BlockVector<double> displacement;
  std::unique_ptr<MappingQEulerian<dim, BlockVector<double>>>
    displacement_mapping;

  DiscreteTime time;

  std::shared_ptr<typename InnerPreconditioner<dim>::type> A_preconditioner;

  friend class RightHandSide<dim>;
};
