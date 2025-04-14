/**
 Thomas Wick
 Leibniz University Hannover
 Institute of Applied Mathematics (IfAM)
 Date: May 22, 2021
 E-mail: thomas.wick@ifam.uni-hannover.de

 Short title: goal-oriented-fsi: Goal-oriented error control for FSI

 Title: Goal-oriented posteriori error estimation and mesh adaptivity
        with a partition-of-unity dual-weighted residual method
        applied to stationary fluid-structure interaction

 Keywords: fluid-structure interaction, nonlinear harmonic MMPDE,
           finite elements, benchmark computation,
     monolithic framework,
     partition-of-unity (PU),
     dual-weighted residual (DWR) a posteriori error estimation


 This code is based on the deal.II.9.1.1 and
 licensed under the "GNU Lesser General Public License (LGPL)"
 with all information in LICENSE.
 Copyright 2017-2021: Thomas Wick


 This code is a modification of
 the ANS (Archive of Numerical Software) article open-source version:

 http://media.archnumsoft.org/10305/

 https://github.com/tommeswick/fsi

 and parts of step-14 of deal.II and the partition-of-unity localization

 T. Richter, T. Wick; Variational Localizations of the Dual-Weighted Residual
 Estimator, Journal of Computational and Applied Mathematics, Vol. 279 (2015),
 pp. 192-208 https://www.sciencedirect.com/science/article/pii/S0377042714004798


 Possible extensions:

 1. Construct symmetric error estimator with primal \rho and adjoint \rho^*
 error parts (see Becker/Rannacher, Acta Numerica, 2001 or Bangerth/Rannacher,
 Book, 2003)

 2. Test other refinement strategies

 3. Other configurations (e.g., driven flow cavity)

 4. Better (cheaper) approximation of dual solution via
    local-higher order interpolations rather than a global higher-order finite
 element

 5. Other goal functionals (currently only drag is implemented)

 6. Overall improvements of linear solvers, e.g., parallelization,
    for better efficiency

 7. 3D

 */



// Include files
//--------------

// The first step, as always, is to include
// the functionality of these
// deal.II library files and some C++ header
// files.
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_direct.h>
// #include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
// From deal.II 9.x.x
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/manifold_lib.h>


// #include <deal.II/lac/constraint_matrix.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>


// C++
#include <fstream>
#include <sstream>

// At the end of this top-matter, we import
// all deal.II names into the global
// namespace:
using namespace dealii;



// First, we define tensors for solution variables
// v (velocity), u (displacement), p (pressure).
// Moreover, we define
// corresponding tensors for derivatives (e.g., gradients,
// deformation gradients) and
// linearized tensors that are needed to solve the
// non-linear problem with Newton's method.
namespace ALE_Transformations
{
  template <int dim>
  inline Tensor<2, dim>
  get_pI(unsigned int q, std::vector<Vector<double>> old_solution_values)
  {
    Tensor<2, dim> tmp;
    tmp[0][0] = old_solution_values[q](dim + dim);
    tmp[1][1] = old_solution_values[q](dim + dim);

    return tmp;
  }

  template <int dim>
  inline Tensor<2, dim>
  get_pI_LinP(const double phi_i_p)
  {
    Tensor<2, dim> tmp;
    tmp.clear();
    tmp[0][0] = phi_i_p;
    tmp[1][1] = phi_i_p;

    return tmp;
  }

  template <int dim>
  inline Tensor<1, dim>
  get_grad_p(unsigned int                             q,
             std::vector<std::vector<Tensor<1, dim>>> old_solution_grads)
  {
    Tensor<1, dim> grad_p;
    grad_p[0] = old_solution_grads[q][dim + dim][0];
    grad_p[1] = old_solution_grads[q][dim + dim][1];

    return grad_p;
  }

  template <int dim>
  inline Tensor<1, dim>
  get_grad_p_LinP(const Tensor<1, dim> phi_i_grad_p)
  {
    Tensor<1, dim> grad_p;
    grad_p[0] = phi_i_grad_p[0];
    grad_p[1] = phi_i_grad_p[1];

    return grad_p;
  }

  template <int dim>
  inline Tensor<2, dim>
  get_grad_u(unsigned int                             q,
             std::vector<std::vector<Tensor<1, dim>>> old_solution_grads)
  {
    Tensor<2, dim> structure_continuation;
    structure_continuation[0][0] = old_solution_grads[q][dim][0];
    structure_continuation[0][1] = old_solution_grads[q][dim][1];
    structure_continuation[1][0] = old_solution_grads[q][dim + 1][0];
    structure_continuation[1][1] = old_solution_grads[q][dim + 1][1];

    return structure_continuation;
  }

  template <int dim>
  inline Tensor<2, dim>
  get_grad_v(unsigned int                             q,
             std::vector<std::vector<Tensor<1, dim>>> old_solution_grads)
  {
    Tensor<2, dim> grad_v;
    grad_v[0][0] = old_solution_grads[q][0][0];
    grad_v[0][1] = old_solution_grads[q][0][1];
    grad_v[1][0] = old_solution_grads[q][1][0];
    grad_v[1][1] = old_solution_grads[q][1][1];

    return grad_v;
  }

  template <int dim>
  inline Tensor<2, dim>
  get_grad_v_T(const Tensor<2, dim> tensor_grad_v)
  {
    Tensor<2, dim> grad_v_T;
    grad_v_T = transpose(tensor_grad_v);

    return grad_v_T;
  }

  template <int dim>
  inline Tensor<2, dim>
  get_grad_v_LinV(const Tensor<2, dim> phi_i_grads_v)
  {
    Tensor<2, dim> tmp;
    tmp[0][0] = phi_i_grads_v[0][0];
    tmp[0][1] = phi_i_grads_v[0][1];
    tmp[1][0] = phi_i_grads_v[1][0];
    tmp[1][1] = phi_i_grads_v[1][1];

    return tmp;
  }

  template <int dim>
  inline Tensor<2, dim>
  get_Identity()
  {
    Tensor<2, dim> identity;
    identity[0][0] = 1.0;
    identity[0][1] = 0.0;
    identity[1][0] = 0.0;
    identity[1][1] = 1.0;

    return identity;
  }

  template <int dim>
  inline Tensor<2, dim>
  get_F(unsigned int                             q,
        std::vector<std::vector<Tensor<1, dim>>> old_solution_grads)
  {
    Tensor<2, dim> F;
    F[0][0] = 1.0 + old_solution_grads[q][dim][0];
    F[0][1] = old_solution_grads[q][dim][1];
    F[1][0] = old_solution_grads[q][dim + 1][0];
    F[1][1] = 1.0 + old_solution_grads[q][dim + 1][1];
    return F;
  }

  template <int dim>
  inline Tensor<2, dim>
  get_F_T(const Tensor<2, dim> F)
  {
    return transpose(F);
  }

  template <int dim>
  inline Tensor<2, dim>
  get_F_Inverse(const Tensor<2, dim> F)
  {
    return invert(F);
  }

  template <int dim>
  inline Tensor<2, dim>
  get_F_Inverse_T(const Tensor<2, dim> F_Inverse)
  {
    return transpose(F_Inverse);
  }

  template <int dim>
  inline double
  get_J(const Tensor<2, dim> tensor_F)
  {
    return determinant(tensor_F);
  }


  template <int dim>
  inline Tensor<1, dim>
  get_v(unsigned int q, std::vector<Vector<double>> old_solution_values)
  {
    Tensor<1, dim> v;
    v[0] = old_solution_values[q](0);
    v[1] = old_solution_values[q](1);

    return v;
  }

  template <int dim>
  inline Tensor<1, dim>
  get_v_LinV(const Tensor<1, dim> phi_i_v)
  {
    Tensor<1, dim> tmp;
    tmp[0] = phi_i_v[0];
    tmp[1] = phi_i_v[1];

    return tmp;
  }

  template <int dim>
  inline Tensor<1, dim>
  get_u(unsigned int q, std::vector<Vector<double>> old_solution_values)
  {
    Tensor<1, dim> u;
    u[0] = old_solution_values[q](dim);
    u[1] = old_solution_values[q](dim + 1);

    return u;
  }

  template <int dim>
  inline Tensor<1, dim>
  get_u_LinU(const Tensor<1, dim> phi_i_u)
  {
    Tensor<1, dim> tmp;
    tmp[0] = phi_i_u[0];
    tmp[1] = phi_i_u[1];

    return tmp;
  }


  template <int dim>
  inline double
  get_J_LinU(unsigned int                                   q,
             const std::vector<std::vector<Tensor<1, dim>>> old_solution_grads,
             const Tensor<2, dim>                           phi_i_grads_u)
  {
    return (phi_i_grads_u[0][0] * (1 + old_solution_grads[q][dim + 1][1]) +
            (1 + old_solution_grads[q][dim][0]) * phi_i_grads_u[1][1] -
            phi_i_grads_u[0][1] * old_solution_grads[q][dim + 1][0] -
            old_solution_grads[q][dim][1] * phi_i_grads_u[1][0]);
  }

  template <int dim>
  inline double
  get_J_Inverse_LinU(const double J, const double J_LinU)
  {
    return (-1.0 / std::pow(J, 2) * J_LinU);
  }

  template <int dim>
  inline Tensor<2, dim>
  get_F_LinU(const Tensor<2, dim> phi_i_grads_u)
  {
    Tensor<2, dim> tmp;
    tmp[0][0] = phi_i_grads_u[0][0];
    tmp[0][1] = phi_i_grads_u[0][1];
    tmp[1][0] = phi_i_grads_u[1][0];
    tmp[1][1] = phi_i_grads_u[1][1];

    return tmp;
  }

  template <int dim>
  inline Tensor<2, dim>
  get_F_Inverse_LinU(
    const Tensor<2, dim>                     phi_i_grads_u,
    const double                             J,
    const double                             J_LinU,
    unsigned int                             q,
    std::vector<std::vector<Tensor<1, dim>>> old_solution_grads)
  {
    Tensor<2, dim> F_tilde;
    F_tilde[0][0] = 1.0 + old_solution_grads[q][dim + 1][1];
    F_tilde[0][1] = -old_solution_grads[q][dim][1];
    F_tilde[1][0] = -old_solution_grads[q][dim + 1][0];
    F_tilde[1][1] = 1.0 + old_solution_grads[q][dim][0];

    Tensor<2, dim> F_tilde_LinU;
    F_tilde_LinU[0][0] = phi_i_grads_u[1][1];
    F_tilde_LinU[0][1] = -phi_i_grads_u[0][1];
    F_tilde_LinU[1][0] = -phi_i_grads_u[1][0];
    F_tilde_LinU[1][1] = phi_i_grads_u[0][0];

    return (-1.0 / (J * J) * J_LinU * F_tilde + 1.0 / J * F_tilde_LinU);
  }

  template <int dim>
  inline Tensor<2, dim>
  get_J_F_Inverse_T_LinU(const Tensor<2, dim> phi_i_grads_u)
  {
    Tensor<2, dim> tmp;
    tmp[0][0] = phi_i_grads_u[1][1];
    tmp[0][1] = -phi_i_grads_u[1][0];
    tmp[1][0] = -phi_i_grads_u[0][1];
    tmp[1][1] = phi_i_grads_u[0][0];

    return tmp;
  }


  template <int dim>
  inline double
  get_tr_C_LinU(
    unsigned int                                   q,
    const std::vector<std::vector<Tensor<1, dim>>> old_solution_grads,
    const Tensor<2, dim>                           phi_i_grads_u)
  {
    return ((1 + old_solution_grads[q][dim][0]) * phi_i_grads_u[0][0] +
            old_solution_grads[q][dim][1] * phi_i_grads_u[0][1] +
            (1 + old_solution_grads[q][dim + 1][1]) * phi_i_grads_u[1][1] +
            old_solution_grads[q][dim + 1][0] * phi_i_grads_u[1][0]);
  }


} // namespace ALE_Transformations

// Second, we define the ALE transformations rules. These
// are used to transform the fluid equations from the Eulerian
// coordinate system to an arbitrary fixed reference
// configuration.
namespace NSE_in_ALE
{
  template <int dim>
  inline Tensor<2, dim>
  get_stress_fluid_ALE(const double         density,
                       const double         viscosity,
                       const Tensor<2, dim> pI,
                       const Tensor<2, dim> grad_v,
                       const Tensor<2, dim> grad_v_T,
                       const Tensor<2, dim> F_Inverse,
                       const Tensor<2, dim> F_Inverse_T)
  {
    return (-pI + density * viscosity *
                    (grad_v * F_Inverse + F_Inverse_T * grad_v_T));
  }

  template <int dim>
  inline Tensor<2, dim>
  get_stress_fluid_except_pressure_ALE(const double         density,
                                       const double         viscosity,
                                       const Tensor<2, dim> grad_v,
                                       const Tensor<2, dim> grad_v_T,
                                       const Tensor<2, dim> F_Inverse,
                                       const Tensor<2, dim> F_Inverse_T)
  {
    return (density * viscosity *
            (grad_v * F_Inverse + F_Inverse_T * grad_v_T));
  }

  template <int dim>
  inline Tensor<2, dim>
  get_stress_fluid_ALE_1st_term_LinAll(const Tensor<2, dim> pI,
                                       const Tensor<2, dim> F_Inverse_T,
                                       const Tensor<2, dim> J_F_Inverse_T_LinU,
                                       const Tensor<2, dim> pI_LinP,
                                       const double         J)
  {
    return (-J * pI_LinP * F_Inverse_T - pI * J_F_Inverse_T_LinU);
  }

  template <int dim>
  inline Tensor<2, dim>
  get_stress_fluid_ALE_2nd_term_LinAll_short(
    const Tensor<2, dim> J_F_Inverse_T_LinU,
    const Tensor<2, dim> stress_fluid_ALE,
    const Tensor<2, dim> grad_v,
    const Tensor<2, dim> grad_v_LinV,
    const Tensor<2, dim> F_Inverse,
    const Tensor<2, dim> F_Inverse_LinU,
    const double         J,
    const double         viscosity,
    const double         density)
  {
    Tensor<2, dim> sigma_LinV;
    Tensor<2, dim> sigma_LinU;

    sigma_LinV =
      grad_v_LinV * F_Inverse + transpose(F_Inverse) * transpose(grad_v_LinV);
    sigma_LinU =
      grad_v * F_Inverse_LinU + transpose(F_Inverse_LinU) * transpose(grad_v);

    return (density * viscosity * (sigma_LinV + sigma_LinU) * J *
              transpose(F_Inverse) +
            stress_fluid_ALE * J_F_Inverse_T_LinU);
  }

  template <int dim>
  inline Tensor<2, dim>
  get_stress_fluid_ALE_3rd_term_LinAll_short(
    const Tensor<2, dim> F_Inverse,
    const Tensor<2, dim> F_Inverse_LinU,
    const Tensor<2, dim> grad_v,
    const Tensor<2, dim> grad_v_LinV,
    const double         viscosity,
    const double         density,
    const double         J,
    const Tensor<2, dim> J_F_Inverse_T_LinU)
  {
    return density * viscosity *
           (J_F_Inverse_T_LinU * transpose(grad_v) * transpose(F_Inverse) +
            J * transpose(F_Inverse) * transpose(grad_v_LinV) *
              transpose(F_Inverse) +
            J * transpose(F_Inverse) * transpose(grad_v) *
              transpose(F_Inverse_LinU));
  }



  template <int dim>
  inline double
  get_Incompressibility_ALE(
    unsigned int                             q,
    std::vector<std::vector<Tensor<1, dim>>> old_solution_grads)
  {
    return (old_solution_grads[q][0][0] +
            old_solution_grads[q][dim + 1][1] * old_solution_grads[q][0][0] -
            old_solution_grads[q][dim][1] * old_solution_grads[q][1][0] -
            old_solution_grads[q][dim + 1][0] * old_solution_grads[q][0][1] +
            old_solution_grads[q][1][1] +
            old_solution_grads[q][dim][0] * old_solution_grads[q][1][1]);
  }

  template <int dim>
  inline double
  get_Incompressibility_ALE_LinAll(
    const Tensor<2, dim>                           phi_i_grads_v,
    const Tensor<2, dim>                           phi_i_grads_u,
    unsigned int                                   q,
    const std::vector<std::vector<Tensor<1, dim>>> old_solution_grads)
  {
    return (phi_i_grads_v[0][0] + phi_i_grads_v[1][1] +
            phi_i_grads_u[1][1] * old_solution_grads[q][0][0] +
            old_solution_grads[q][dim + 1][1] * phi_i_grads_v[0][0] -
            phi_i_grads_u[0][1] * old_solution_grads[q][1][0] -
            old_solution_grads[q][dim + 0][1] * phi_i_grads_v[1][0] -
            phi_i_grads_u[1][0] * old_solution_grads[q][0][1] -
            old_solution_grads[q][dim + 1][0] * phi_i_grads_v[0][1] +
            phi_i_grads_u[0][0] * old_solution_grads[q][1][1] +
            old_solution_grads[q][dim + 0][0] * phi_i_grads_v[1][1]);
  }


  template <int dim>
  inline Tensor<1, dim>
  get_Convection_LinAll_short(const Tensor<2, dim> phi_i_grads_v,
                              const Tensor<1, dim> phi_i_v,
                              const double         J,
                              const double         J_LinU,
                              const Tensor<2, dim> F_Inverse,
                              const Tensor<2, dim> F_Inverse_LinU,
                              const Tensor<1, dim> v,
                              const Tensor<2, dim> grad_v,
                              const double         density)
  {
    // Linearization of fluid convection term
    // rho J(F^{-1}v\cdot\grad)v = rho J grad(v)F^{-1}v

    Tensor<1, dim> convection_LinU;
    convection_LinU =
      (J_LinU * grad_v * F_Inverse * v + J * grad_v * F_Inverse_LinU * v);

    Tensor<1, dim> convection_LinV;
    convection_LinV =
      (J * (phi_i_grads_v * F_Inverse * v + grad_v * F_Inverse * phi_i_v));

    return density * (convection_LinU + convection_LinV);
  }


  template <int dim>
  inline Tensor<1, dim>
  get_Convection_u_LinAll_short(const Tensor<2, dim> phi_i_grads_v,
                                const Tensor<1, dim> phi_i_u,
                                const double         J,
                                const double         J_LinU,
                                const Tensor<2, dim> F_Inverse,
                                const Tensor<2, dim> F_Inverse_LinU,
                                const Tensor<1, dim> u,
                                const Tensor<2, dim> grad_v,
                                const double         density)
  {
    // Linearization of fluid convection term
    // rho J(F^{-1}v\cdot\grad)u = rho J grad(v)F^{-1}u

    Tensor<1, dim> convection_LinU;
    convection_LinU =
      (J_LinU * grad_v * F_Inverse * u + J * grad_v * F_Inverse_LinU * u +
       J * grad_v * F_Inverse * phi_i_u);

    Tensor<1, dim> convection_LinV;
    convection_LinV = (J * phi_i_grads_v * F_Inverse * u);

    return density * (convection_LinU + convection_LinV);
  }



  template <int dim>
  inline Tensor<1, dim>
  get_Convection_u_old_LinAll_short(
    const Tensor<2, dim> phi_i_grads_v,
    const double         J,
    const double         J_LinU,
    const Tensor<2, dim> F_Inverse,
    const Tensor<2, dim> F_Inverse_LinU,
    const Tensor<1, dim> old_timestep_solution_displacement,
    const Tensor<2, dim> grad_v,
    const double         density)
  {
    // Linearization of fluid convection term
    // rho J(F^{-1}v\cdot\grad)u = rho J grad(v)F^{-1}u

    Tensor<1, dim> convection_LinU;
    convection_LinU =
      (J_LinU * grad_v * F_Inverse * old_timestep_solution_displacement +
       J * grad_v * F_Inverse_LinU * old_timestep_solution_displacement);

    Tensor<1, dim> convection_LinV;
    convection_LinV =
      (J * phi_i_grads_v * F_Inverse * old_timestep_solution_displacement);


    return density * (convection_LinU + convection_LinV);
  }

  template <int dim>
  inline Tensor<1, dim>
  get_accelaration_term_LinAll(const Tensor<1, dim> phi_i_v,
                               const Tensor<1, dim> v,
                               const Tensor<1, dim> old_timestep_v,
                               const double         J_LinU,
                               const double         J,
                               const double         old_timestep_J,
                               const double         density)
  {
    return density / 2.0 *
           (J_LinU * (v - old_timestep_v) + (J + old_timestep_J) * phi_i_v);
  }


} // namespace NSE_in_ALE


// In the third namespace, we summarize the
// constitutive relations for the solid equations.
namespace Structure_Terms_in_ALE
{
  // Green-Lagrange strain tensor
  template <int dim>
  inline Tensor<2, dim>
  get_E(const Tensor<2, dim> F_T,
        const Tensor<2, dim> F,
        const Tensor<2, dim> Identity)
  {
    return 0.5 * (F_T * F - Identity);
  }

  template <int dim>
  inline double
  get_tr_E(const Tensor<2, dim> E)
  {
    return trace(E);
  }

  template <int dim>
  inline double
  get_tr_E_LinU(
    unsigned int                                   q,
    const std::vector<std::vector<Tensor<1, dim>>> old_solution_grads,
    const Tensor<2, dim>                           phi_i_grads_u)
  {
    return ((1 + old_solution_grads[q][dim][0]) * phi_i_grads_u[0][0] +
            old_solution_grads[q][dim][1] * phi_i_grads_u[0][1] +
            (1 + old_solution_grads[q][dim + 1][1]) * phi_i_grads_u[1][1] +
            old_solution_grads[q][dim + 1][0] * phi_i_grads_u[1][0]);
  }

} // namespace Structure_Terms_in_ALE



// In this class, we define a function
// that deals with the boundary values.
// For our configuration,
// we impose of parabolic inflow profile for the
// velocity at the left hand side of the channel.
template <int dim>
class BoundaryParabel : public Function<dim>
{
public:
  BoundaryParabel(const double time)
    : Function<dim>(dim + dim + 1)
  {
    _time = time;
  }

  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const;

  virtual void
  vector_value(const Point<dim> &p, Vector<double> &value) const;

private:
  double _time;
};

// The boundary values are given to component
// with number 0 (namely the x-velocity)
template <int dim>
double
BoundaryParabel<dim>::value(const Point<dim>  &p,
                            const unsigned int component) const
{
  Assert(component < this->n_components,
         ExcIndexRange(component, 0, this->n_components));

  // The maximum inflow depends on the configuration
  // for the different test cases:
  // FSI 1: 0.2;
  // FSI 2: 1.0;
  // FSI 3: 2.0;
  //
  // For the two unsteady test cases FSI 2 and FSI 3, it
  // is recommanded to start with a smooth increase of
  // the inflow. Hence, we use the cosine function
  // to control the inflow at the beginning until
  // the total time 2.0 has been reached.
  double inflow_velocity = 0.2;

  if (component == 0)
    {
      // FSI 1 and BFAC 2D-1, 2D-2
      {
        return ((p(0) == 0) && (p(1) <= 0.41) ?
                  -1.5 * inflow_velocity * (4.0 / 0.1681) *
                    (std::pow(p(1), 2) - 0.41 * std::pow(p(1), 1)) :
                  0);
      }
    }

  return 0;
}



template <int dim>
void
BoundaryParabel<dim>::vector_value(const Point<dim> &p,
                                   Vector<double>   &values) const
{
  for (unsigned int c = 0; c < this->n_components; ++c)
    values(c) = BoundaryParabel<dim>::value(p, c);
}


// In the next class, we define the main problem at hand.
// Here, we implement
// the top-level logic of solving a
// stationary FSI problem in a
// variational-monolithic ALE framework.
//
// The initial framework of our program is based on the
// deal.II step-22 and step-14 tutorial programs and the
// FSI ANS-step (T. Wick; Archive for Numerical Software, 2013)
// based on harmonic mesh motion. Step-22
// explains best how to deal with vector-valued problems in
// deal.II. We extend that program by several additional elements:
//
//
// i)   additional non-linearity in the fluid (convection term)
//      -> requires non-linear solution algorithm
// ii)  non-linear structure problem that is fully coupled to the fluid
//      -> second source of non-linearities due to the transformation
// iii) implementation of a Newton-like method to solve the non-linear problem
//
// iv) Goal-oriented mesh refinement using
//     a partition-of-unity dual-weighted residual method.
//
// To construct the ALE mapping for the fluid mesh motion, we
// solve an additional partial differential equation that
// is given by a nonlinear harmonic equation.
//
// All equations are written in a common global system that
// is referred to as a variational-monolithic solution algorithm.
//
//
// The  program is organized as follows. First, we set up
// runtime parameters and the system as done in other deal.II tutorial steps.
// Then, we assemble
// the system matrix (Jacobian of Newton's method)
// and system right hand side (residual of Newton's method) for the non-linear
// system. Two functions for the boundary values are provided because
// we are only supposed to apply boundary values in the first Newton step. In
// the subsequent Newton steps all Dirichlet values have to be equal zero.
// Afterwards, the routines for solving the linear
// system and the Newton iteration are self-explaining. The following
// function is standard in deal.II tutorial steps:
// writing the solutions to graphical output.
// The last three functions provide the framework to compute
// functional values of interest. For the given fluid-structure
// interaction problem, we compute the displacement in the x- and y-directions
// of the structure at a certain point. We are also interested in the
// observation of the drag- and lift evaluations, which are achieved by
// line-integration over faces or alternatively via domain integration.
template <int dim>
class FSI_PU_DWR_Problem
{
public:
  FSI_PU_DWR_Problem(const unsigned int degree);
  ~FSI_PU_DWR_Problem();
  void
  run();

private:
  // Setup of material parameters,
  // spatial grid, etc. for primal and adjoint problems
  void
  set_runtime_parameters();


  // Primal
  // Create system matrix, rhs and distribute degrees of freedom.
  void
  setup_system_primal();

  // Assemble left and right hand side for Newton's method
  void
  assemble_matrix_primal();
  void
  assemble_rhs_primal();

  // Boundary conditions (bc)
  void
  set_initial_bc_primal();
  void
  set_newton_bc_primal();

  // Linear primal solver
  void
  solve_primal();

  // Nonlinear primal solver
  void
  newton_iteration_primal();


  //// Adjoint problem
  void
  setup_system_adjoint();
  void
  assemble_matrix_adjoint();
  void
  assemble_rhs_adjoint_drag();
  void
  assemble_rhs_adjoint_pressure_point();
  void
  assemble_rhs_adjoint_displacement_point();
  void
  set_bc_adjoint();
  // only linear solver because adjoint is linear
  void
  solve_adjoint();



  // Graphical visualization of output
  void
  output_results(const unsigned int refinement_cycle) const;


  // Evaluation of functional values
  double
  compute_point_value(Point<dim> p, const unsigned int component) const;

  void
  compute_drag_lift_fsi_fluid_tensor();
  void
  compute_drag_lift_fsi_fluid_tensor_domain();
  void
  compute_drag_lift_fsi_fluid_tensor_domain_structure();

  void
  compute_functional_values();
  void
  compute_minimal_J();

  // Local mesh refinement
  void
  refine_mesh();
  double
  compute_error_indicators_a_la_PU_DWR(const unsigned int refinement_cycle);
  double
  refine_average_with_PU_DWR(const unsigned int refinement_cycle);


  // Global refinement
  const unsigned int degree;

  Triangulation<dim> triangulation;

  // Primal solution
  FESystem<dim>   fe_primal;
  DoFHandler<dim> dof_handler_primal;

  ConstraintMatrix constraints_primal;

  BlockSparsityPattern      sparsity_pattern_primal;
  BlockSparseMatrix<double> system_matrix_primal;

  BlockVector<double> solution_primal, newton_update_primal,
    old_timestep_solution_primal;
  BlockVector<double> system_rhs_primal;

  SparseDirectUMFPACK A_direct_primal;

  // Adjoint solution
  FESystem<dim>   fe_adjoint;
  DoFHandler<dim> dof_handler_adjoint;

  ConstraintMatrix constraints_adjoint;

  BlockSparsityPattern      sparsity_pattern_adjoint;
  BlockSparseMatrix<double> system_matrix_adjoint;

  BlockVector<double> solution_adjoint, old_timestep_solution_adjoint;
  BlockVector<double> system_rhs_adjoint;

  // PU for PU-DWR localization
  FESystem<dim>   fe_pou;
  DoFHandler<dim> dof_handler_pou;
  Vector<float>   error_indicators;


  // Measuring CPU times
  TimerOutput timer;


  // Fluid parameters
  double density_fluid, viscosity;

  // Structure parameters
  double density_structure;
  double lame_coefficient_mu, lame_coefficient_lambda, poisson_ratio_nu;

  // Other parameters to control the fluid mesh motion
  double cell_diameter;
  double alpha_u;

  // Right hand side forces and values
  double force_structure_x, force_structure_y, force_fluid_x, force_fluid_y;

  double global_drag_lift_value;

  std::string   test_case, adjoint_rhs;
  std::ofstream file_gnuplot;
  std::ofstream file;

  unsigned int max_no_refinement_cycles, max_no_degrees_of_freedom;
  double       TOL_DWR_estimator, lower_bound_newton_residual,
    lower_bound_newton_residual_relative;

  unsigned int refinement_strategy;

  double reference_value_drag, reference_value_lift, reference_value_p_front,
    reference_value_p_diff, exact_error_local, reference_value_flag_tip_ux,
    reference_value_flag_tip_uy;
};


// The constructor of this class is comparable
// to other tutorials steps, e.g., step-22, and step-31.
// We are going to use the following finite element discretization:
// Q_2^c for the fluid, Q_2^c for the solid, P_1^dc for the pressure.
template <int dim>
FSI_PU_DWR_Problem<dim>::FSI_PU_DWR_Problem(const unsigned int degree)
  : degree(degree)
  ,
  // triangulation (Triangulation<dim>::maximum_smoothing),
  triangulation(Triangulation<dim>::none)
  ,

  fe_primal(FE_Q<dim>(2),
            dim, // velocities
            FE_Q<dim>(2),
            dim, // displacements
            FE_Q<dim>(1),
            1)
  , // pressure
  dof_handler_primal(triangulation)
  ,

  // Info: adjoint degree must be higher than primal
  // degree as usual for DWR-based error estimation (see step-14)
  fe_adjoint(FE_Q<dim>(4),
             dim, // velocities
             FE_Q<dim>(4),
             dim, // displacements
             FE_Q<dim>(2),
             1)
  , // pressure
  dof_handler_adjoint(triangulation)
  ,

  // Lowest order FE (partition-of-unity) to gather neighboring information
  // see Richter/Wick; JCAM, 2015
  // https://www.sciencedirect.com/science/article/pii/S0377042714004798
  fe_pou(FE_Q<dim>(1), 1)
  , dof_handler_pou(triangulation)
  ,

  timer(std::cout, TimerOutput::summary, TimerOutput::cpu_times)
{}


// This is the standard destructor.
template <int dim>
FSI_PU_DWR_Problem<dim>::~FSI_PU_DWR_Problem()
{}


// In this method, we set up runtime parameters that
// could also come from a paramter file. We propose
// three different configurations FSI 1, FSI 2, and FSI 3.
// The reader is invited to change these values to obtain
// other results.
template <int dim>
void
FSI_PU_DWR_Problem<dim>::set_runtime_parameters()
{
  // Defining test cases
  // 2D-1: Schaefer/Turek benchmark 1996
  // FSI_1: Hron/Turek benchmark 2006
  test_case = "FSI_1";
  // pressure, drag, lift, displacement
  adjoint_rhs               = "drag";
  max_no_refinement_cycles  = 4;
  max_no_degrees_of_freedom = 2.0e+6;
  TOL_DWR_estimator         = 1.0e-5;

  // Setting tolerance for primal Newton solver
  lower_bound_newton_residual          = 1.0e-10;
  lower_bound_newton_residual_relative = 1.0e-8;

  // 0: global refinement
  // 1: PU DWR
  // 2: Solid-oriented
  refinement_strategy = 1;

  // Fluid parameters (FSI 1)
  if (test_case == "FSI_1")
    {
      density_fluid = 1.0e+3;
      viscosity     = 1.0e-3;
    }

  // 2D-1 1996 benchmark
  if (test_case == "2D-1")
    {
      density_fluid = 1.0;
      viscosity     = 1.0e-3;
    }

  // FSI 1 & 3: 1.0e+3; FSI 2: 1.0e+4
  density_structure = 1.0e+3;


  // Structure parameters
  // FSI 1 & 2: 0.5e+6; FSI 3: 2.0e+6
  lame_coefficient_mu = 0.5e+6;
  poisson_ratio_nu    = 0.4;

  lame_coefficient_lambda =
    (2 * poisson_ratio_nu * lame_coefficient_mu) / (1.0 - 2 * poisson_ratio_nu);

  // Force on beam
  force_structure_x = 0.0;
  force_structure_y = 0.0;

  // For example gravity
  force_fluid_x = 0.0;
  force_fluid_y = 0.0;


  // Diffusion parameters to control the fluid mesh motion
  // The higher these parameters the stiffer the fluid mesh.
  alpha_u = 1.0e-10;


  // In the following, we read a *.inp grid from a file.
  // The geometry information is based on the
  // fluid-structure interaction benchmark problems
  // (Lit. J. Hron, S. Turek, 2006)
  std::string grid_name;
  if (test_case == "FSI_1")
    grid_name = "fsi.inp";
  else if (test_case == "2D-1")
    grid_name = "nsbench4_original.inp";

  GridIn<dim> grid_in;
  grid_in.attach_triangulation(triangulation);
  std::ifstream input_file(grid_name.c_str());
  Assert(dim == 2, ExcInternalError());
  grid_in.read_ucd(input_file);

  Point<dim> p(0.2, 0.2);
  // double radius = 0.05;
  // static const HyperBallBoundary<dim> boundary(p,radius);
  // triangulation.set_boundary (80, boundary);
  // triangulation.set_boundary (81, boundary);

  // from deal_II_version 9.0.0
  static const dealii::SphericalManifold<dim> boundary(p);
  triangulation.reset_all_manifolds();
  triangulation.set_all_manifold_ids_on_boundary(80, 80);
  triangulation.set_all_manifold_ids_on_boundary(81, 81);
  triangulation.set_manifold(80, boundary);
  triangulation.set_manifold(81, boundary);

  triangulation.refine_global(1);

  // For DWR output (effectivity indices, error behavior, etc.) into files
  std::string filename         = "dwr_results.txt";
  std::string filename_gnuplot = "dwr_results_gp.txt";
  file.open(filename.c_str());
  file_gnuplot.open(filename_gnuplot.c_str());

  file << "Dofs" << "\t" << "Exact err" << "\t" << "Est err   " << "\t"
       << "Est ind   " << "\t" << "Eff" << "\t" << "Ind" << "\n";
  file.flush();
}



// This function is similar to many deal.II tuturial steps.
template <int dim>
void
FSI_PU_DWR_Problem<dim>::setup_system_primal()
{
  timer.enter_section("Setup system.");

  system_matrix_primal.clear();

  dof_handler_primal.distribute_dofs(fe_primal);
  DoFRenumbering::Cuthill_McKee(dof_handler_primal);

  // We are dealing with 7 components for this
  // two-dimensional fluid-structure interacion problem
  // Precisely, we use:
  // velocity in x and y:                0
  // structure displacement in x and y:  1
  // scalar pressure field:              2
  std::vector<unsigned int> block_component(5, 0);
  block_component[dim]       = 1;
  block_component[dim + 1]   = 1;
  block_component[dim + dim] = 2;

  DoFRenumbering::component_wise(dof_handler_primal, block_component);

  {
    constraints_primal.clear();
    set_newton_bc_primal();
    DoFTools::make_hanging_node_constraints(dof_handler_primal,
                                            constraints_primal);
  }
  constraints_primal.close();

  std::vector<types::global_dof_index> dofs_per_block(3);
  DoFTools::count_dofs_per_block(dof_handler_primal,
                                 dofs_per_block,
                                 block_component);
  const unsigned int n_v = dofs_per_block[0], n_u = dofs_per_block[1],
                     n_p = dofs_per_block[2];

  std::cout << "Cells:\t" << triangulation.n_active_cells() << std::endl
            << "DoFs (primal):\t" << dof_handler_primal.n_dofs() << " (" << n_v
            << '+' << n_u << '+' << n_p << ')' << std::endl;



  {
    BlockDynamicSparsityPattern csp(3, 3);

    csp.block(0, 0).reinit(n_v, n_v);
    csp.block(0, 1).reinit(n_v, n_u);
    csp.block(0, 2).reinit(n_v, n_p);

    csp.block(1, 0).reinit(n_u, n_v);
    csp.block(1, 1).reinit(n_u, n_u);
    csp.block(1, 2).reinit(n_u, n_p);

    csp.block(2, 0).reinit(n_p, n_v);
    csp.block(2, 1).reinit(n_p, n_u);
    csp.block(2, 2).reinit(n_p, n_p);

    csp.collect_sizes();


    DoFTools::make_sparsity_pattern(dof_handler_primal,
                                    csp,
                                    constraints_primal,
                                    false);

    sparsity_pattern_primal.copy_from(csp);
  }

  system_matrix_primal.reinit(sparsity_pattern_primal);

  // Actual solution
  solution_primal.reinit(3);
  solution_primal.block(0).reinit(n_v);
  solution_primal.block(1).reinit(n_u);
  solution_primal.block(2).reinit(n_p);

  solution_primal.collect_sizes();

  // Updates for Newton's method
  newton_update_primal.reinit(3);
  newton_update_primal.block(0).reinit(n_v);
  newton_update_primal.block(1).reinit(n_u);
  newton_update_primal.block(2).reinit(n_p);

  newton_update_primal.collect_sizes();

  // Residual for  Newton's method
  system_rhs_primal.reinit(3);
  system_rhs_primal.block(0).reinit(n_v);
  system_rhs_primal.block(1).reinit(n_u);
  system_rhs_primal.block(2).reinit(n_p);

  system_rhs_primal.collect_sizes();

  timer.exit_section();
}


// In this function, we assemble the Jacobian matrix
// for the Newton iteration. The fluid and the structure
// equations are computed on different sub-domains
// in the mesh and ask for the corresponding
// material ids. The fluid equations are defined on
// mesh cells with the material id == 0 and the structure
// equations on cells with the material id == 1.
//
// To compensate the well-known problem in fluid
// dynamics on the outflow boundary, we also
// add some correction term on the outflow boundary.
// This relation is known as `do-nothing' condition.
// In the inner loops of the local_cell_matrix.
//
// Assembling of the inner most loop is treated with help of
// the fe.system_to_component_index(j).first function from
// the library.
// Using this function makes the assembling process much faster
// than running over all local degrees of freedom.
template <int dim>
void
FSI_PU_DWR_Problem<dim>::assemble_matrix_primal()
{
  timer.enter_section("Assemble primal matrix.");
  system_matrix_primal = 0;

  QGauss<dim>     quadrature_formula(degree + 2);
  QGauss<dim - 1> face_quadrature_formula(degree + 2);

  FEValues<dim> fe_values(fe_primal,
                          quadrature_formula,
                          update_values | update_quadrature_points |
                            update_JxW_values | update_gradients);

  FEFaceValues<dim> fe_face_values(fe_primal,
                                   face_quadrature_formula,
                                   update_values | update_quadrature_points |
                                     update_normal_vectors | update_gradients |
                                     update_JxW_values);

  const unsigned int dofs_per_cell = fe_primal.dofs_per_cell;

  const unsigned int n_q_points      = quadrature_formula.size();
  const unsigned int n_face_q_points = face_quadrature_formula.size();

  FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);


  // Now, we are going to use the
  // FEValuesExtractors to determine
  // the four principle variables
  const FEValuesExtractors::Vector velocities(0);       // 0
  const FEValuesExtractors::Vector displacements(dim);  // 2
  const FEValuesExtractors::Scalar pressure(dim + dim); // 4

  // We declare Vectors and Tensors for
  // the solutions at the previous Newton iteration:
  std::vector<Vector<double>> old_solution_values(n_q_points,
                                                  Vector<double>(dim + dim +
                                                                 1));

  std::vector<std::vector<Tensor<1, dim>>> old_solution_grads(
    n_q_points, std::vector<Tensor<1, dim>>(dim + dim + 1));

  std::vector<Vector<double>> old_solution_face_values(n_face_q_points,
                                                       Vector<double>(dim +
                                                                      dim + 1));

  std::vector<std::vector<Tensor<1, dim>>> old_solution_face_grads(
    n_face_q_points, std::vector<Tensor<1, dim>>(dim + dim + 1));


  // Declaring test functions:
  std::vector<Tensor<1, dim>> phi_i_v(dofs_per_cell);
  std::vector<Tensor<2, dim>> phi_i_grads_v(dofs_per_cell);
  std::vector<double>         phi_i_p(dofs_per_cell);
  std::vector<Tensor<1, dim>> phi_i_grads_p(dofs_per_cell);
  std::vector<Tensor<1, dim>> phi_i_u(dofs_per_cell);
  std::vector<Tensor<2, dim>> phi_i_grads_u(dofs_per_cell);

  // This is the identity matrix in two dimensions:
  const Tensor<2, dim> Identity = ALE_Transformations ::get_Identity<dim>();

  typename DoFHandler<dim>::active_cell_iterator cell = dof_handler_primal
                                                          .begin_active(),
                                                 endc =
                                                   dof_handler_primal.end();

  for (; cell != endc; ++cell)
    {
      fe_values.reinit(cell);
      local_matrix = 0;

      // We need the cell diameter to control the fluid mesh motion
      cell_diameter = cell->diameter();

      // Old Newton iteration values
      fe_values.get_function_values(solution_primal, old_solution_values);
      fe_values.get_function_gradients(solution_primal, old_solution_grads);


      // Next, we run over all cells for the fluid equations
      if (cell->material_id() == 0)
        {
          for (unsigned int q = 0; q < n_q_points; ++q)
            {
              for (unsigned int k = 0; k < dofs_per_cell; ++k)
                {
                  phi_i_v[k]       = fe_values[velocities].value(k, q);
                  phi_i_grads_v[k] = fe_values[velocities].gradient(k, q);
                  phi_i_p[k]       = fe_values[pressure].value(k, q);
                  phi_i_u[k]       = fe_values[displacements].value(k, q);
                  phi_i_grads_u[k] = fe_values[displacements].gradient(k, q);
                }

              // We build values, vectors, and tensors
              // from information of the previous Newton step. These are
              // introduced for two reasons: First, these are used to perform
              // the ALE mapping of the fluid equations. Second, these terms are
              // used to make the notation as simple and self-explaining as
              // possible:
              const Tensor<2, dim> pI =
                ALE_Transformations ::get_pI<dim>(q, old_solution_values);

              const Tensor<1, dim> v =
                ALE_Transformations ::get_v<dim>(q, old_solution_values);

              const Tensor<2, dim> grad_v =
                ALE_Transformations ::get_grad_v<dim>(q, old_solution_grads);

              const Tensor<2, dim> grad_v_T =
                ALE_Transformations ::get_grad_v_T<dim>(grad_v);

              //	      const Tensor<2,dim> grad_u = ALE_Transformations
              //	::get_grad_u<dim> (q, old_solution_grads);

              const Tensor<2, dim> F =
                ALE_Transformations ::get_F<dim>(q, old_solution_grads);

              const Tensor<2, dim> F_Inverse =
                ALE_Transformations ::get_F_Inverse<dim>(F);

              const Tensor<2, dim> F_Inverse_T =
                ALE_Transformations ::get_F_Inverse_T<dim>(F_Inverse);

              const double J = ALE_Transformations ::get_J<dim>(F);


              // Stress tensor for the fluid in ALE notation
              const Tensor<2, dim> sigma_ALE =
                NSE_in_ALE ::get_stress_fluid_ALE<dim>(density_fluid,
                                                       viscosity,
                                                       pI,
                                                       grad_v,
                                                       grad_v_T,
                                                       F_Inverse,
                                                       F_Inverse_T);

              // Outer loop for dofs
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  const Tensor<2, dim> pI_LinP =
                    ALE_Transformations ::get_pI_LinP<dim>(phi_i_p[i]);

                  const Tensor<2, dim> grad_v_LinV =
                    ALE_Transformations ::get_grad_v_LinV<dim>(
                      phi_i_grads_v[i]);

                  const double J_LinU =
                    ALE_Transformations ::get_J_LinU<dim>(q,
                                                          old_solution_grads,
                                                          phi_i_grads_u[i]);

                  const Tensor<2, dim> J_F_Inverse_T_LinU =
                    ALE_Transformations ::get_J_F_Inverse_T_LinU<dim>(
                      phi_i_grads_u[i]);

                  const Tensor<2, dim> F_Inverse_LinU =
                    ALE_Transformations ::get_F_Inverse_LinU(
                      phi_i_grads_u[i], J, J_LinU, q, old_solution_grads);

                  const Tensor<2, dim> stress_fluid_ALE_1st_term_LinAll =
                    NSE_in_ALE ::get_stress_fluid_ALE_1st_term_LinAll<dim>(
                      pI, F_Inverse_T, J_F_Inverse_T_LinU, pI_LinP, J);

                  const Tensor<2, dim> stress_fluid_ALE_2nd_term_LinAll =
                    NSE_in_ALE ::get_stress_fluid_ALE_2nd_term_LinAll_short(
                      J_F_Inverse_T_LinU,
                      sigma_ALE,
                      grad_v,
                      grad_v_LinV,
                      F_Inverse,
                      F_Inverse_LinU,
                      J,
                      viscosity,
                      density_fluid);

                  const Tensor<1, dim> convection_fluid_LinAll_short =
                    NSE_in_ALE ::get_Convection_LinAll_short<dim>(
                      phi_i_grads_v[i],
                      phi_i_v[i],
                      J,
                      J_LinU,
                      F_Inverse,
                      F_Inverse_LinU,
                      v,
                      grad_v,
                      density_fluid);

                  const double incompressibility_ALE_LinAll =
                    NSE_in_ALE ::get_Incompressibility_ALE_LinAll<dim>(
                      phi_i_grads_v[i],
                      phi_i_grads_u[i],
                      q,
                      old_solution_grads);



                  // Inner loop for dofs
                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                      // Fluid , NSE in ALE
                      const unsigned int comp_j =
                        fe_primal.system_to_component_index(j).first;
                      if (comp_j == 0 || comp_j == 1)
                        {
                          local_matrix(j, i) +=
                            (convection_fluid_LinAll_short * phi_i_v[j] +
                             scalar_product(stress_fluid_ALE_1st_term_LinAll,
                                            phi_i_grads_v[j]) +
                             scalar_product(stress_fluid_ALE_2nd_term_LinAll,
                                            phi_i_grads_v[j])) *
                            fe_values.JxW(q);
                        }
                      else if (comp_j == 2 || comp_j == 3)
                        {
                          // Harmonic MMPDE
                          local_matrix(j, i) +=
                            (alpha_u * scalar_product(phi_i_grads_u[i],
                                                      phi_i_grads_u[j])) *
                            fe_values.JxW(q);
                        }
                      else if (comp_j == 4)
                        {
                          local_matrix(j, i) +=
                            (incompressibility_ALE_LinAll * phi_i_p[j]) *
                            fe_values.JxW(q);
                        }
                      // end j dofs
                    }
                  // end i dofs
                }
              // end n_q_points
            }

          // We compute in the following
          // one term on the outflow boundary.
          // This relation is well-know in the literature
          // as "do-nothing" condition (Heywood/Rannacher/Turek, 1996).
          // Therefore, we only ask for the corresponding color at the outflow
          // boundary that is 1 in our case.
          for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
               ++face)
            {
              if (cell->face(face)->at_boundary() &&
                  (cell->face(face)->boundary_id() == 1))
                {
                  fe_face_values.reinit(cell, face);

                  fe_face_values.get_function_values(solution_primal,
                                                     old_solution_face_values);
                  fe_face_values.get_function_gradients(
                    solution_primal, old_solution_face_grads);

                  for (unsigned int q = 0; q < n_face_q_points; ++q)
                    {
                      for (unsigned int k = 0; k < dofs_per_cell; ++k)
                        {
                          phi_i_v[k] = fe_face_values[velocities].value(k, q);
                          phi_i_grads_v[k] =
                            fe_face_values[velocities].gradient(k, q);
                          phi_i_grads_u[k] =
                            fe_face_values[displacements].gradient(k, q);
                        }

                      const Tensor<2, dim> grad_v =
                        ALE_Transformations ::get_grad_v<dim>(
                          q, old_solution_face_grads);

                      const Tensor<2, dim> F = ALE_Transformations ::get_F<dim>(
                        q, old_solution_face_grads);

                      const Tensor<2, dim> F_Inverse =
                        ALE_Transformations ::get_F_Inverse<dim>(F);

                      const double J = ALE_Transformations ::get_J<dim>(F);


                      for (unsigned int i = 0; i < dofs_per_cell; ++i)
                        {
                          const Tensor<2, dim> grad_v_LinV =
                            ALE_Transformations ::get_grad_v_LinV<dim>(
                              phi_i_grads_v[i]);

                          const double J_LinU =
                            ALE_Transformations ::get_J_LinU<dim>(
                              q, old_solution_face_grads, phi_i_grads_u[i]);

                          const Tensor<2, dim> J_F_Inverse_T_LinU =
                            ALE_Transformations ::get_J_F_Inverse_T_LinU<dim>(
                              phi_i_grads_u[i]);

                          const Tensor<2, dim> F_Inverse_LinU =
                            ALE_Transformations ::get_F_Inverse_LinU(
                              phi_i_grads_u[i],
                              J,
                              J_LinU,
                              q,
                              old_solution_face_grads);

                          const Tensor<2, dim>
                            stress_fluid_ALE_3rd_term_LinAll = NSE_in_ALE ::
                              get_stress_fluid_ALE_3rd_term_LinAll_short<dim>(
                                F_Inverse,
                                F_Inverse_LinU,
                                grad_v,
                                grad_v_LinV,
                                viscosity,
                                density_fluid,
                                J,
                                J_F_Inverse_T_LinU);

                          // Here, we multiply the symmetric part of fluid's
                          // stress tensor with the normal direction.
                          const Tensor<1, dim> neumann_value =
                            (stress_fluid_ALE_3rd_term_LinAll *
                             fe_face_values.normal_vector(q));

                          for (unsigned int j = 0; j < dofs_per_cell; ++j)
                            {
                              const unsigned int comp_j =
                                fe_primal.system_to_component_index(j).first;
                              if (comp_j == 0 || comp_j == 1)
                                {
                                  local_matrix(j, i) -= neumann_value *
                                                        phi_i_v[j] *
                                                        fe_face_values.JxW(q);
                                }

                            } // end j

                        } // end i

                    } // end q_face_points

                } // end if-routine face integrals

            } // end face integrals do-nothing


          // This is the same as discussed in step-22:
          cell->get_dof_indices(local_dof_indices);
          constraints_primal.distribute_local_to_global(local_matrix,
                                                        local_dof_indices,
                                                        system_matrix_primal);

          // Finally, we arrive at the end for assembling the matrix
          // for the fluid equations and step to the computation of the
          // structure terms:
        }
      else if (cell->material_id() == 1)
        {
          for (unsigned int q = 0; q < n_q_points; ++q)
            {
              for (unsigned int k = 0; k < dofs_per_cell; ++k)
                {
                  phi_i_v[k]       = fe_values[velocities].value(k, q);
                  phi_i_grads_v[k] = fe_values[velocities].gradient(k, q);
                  phi_i_p[k]       = fe_values[pressure].value(k, q);
                  phi_i_grads_p[k] = fe_values[pressure].gradient(k, q);
                  phi_i_u[k]       = fe_values[displacements].value(k, q);
                  phi_i_grads_u[k] = fe_values[displacements].gradient(k, q);
                }

              // It is here the same as already shown for the fluid equations.
              // First, we prepare things coming from the previous Newton
              // iteration...
              const Tensor<2, dim> F =
                ALE_Transformations ::get_F<dim>(q, old_solution_grads);

              const Tensor<2, dim> F_T = ALE_Transformations ::get_F_T<dim>(F);


              const Tensor<2, dim> E =
                Structure_Terms_in_ALE ::get_E<dim>(F_T, F, Identity);

              const double tr_E = Structure_Terms_in_ALE ::get_tr_E<dim>(E);


              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  const Tensor<2, dim> F_LinU =
                    ALE_Transformations ::get_F_LinU<dim>(phi_i_grads_u[i]);


                  // STVK: Green-Lagrange strain tensor derivatives
                  const Tensor<2, dim> E_LinU =
                    0.5 * (transpose(F_LinU) * F + transpose(F) * F_LinU);

                  const double tr_E_LinU =
                    Structure_Terms_in_ALE ::get_tr_E_LinU<dim>(
                      q, old_solution_grads, phi_i_grads_u[i]);


                  // STVK
                  // Piola-kirchhoff stress structure STVK linearized in all
                  // directions
                  Tensor<2, dim> piola_kirchhoff_stress_structure_STVK_LinALL;
                  piola_kirchhoff_stress_structure_STVK_LinALL =
                    lame_coefficient_lambda *
                      (F_LinU * tr_E * Identity + F * tr_E_LinU * Identity) +
                    2 * lame_coefficient_mu * (F_LinU * E + F * E_LinU);


                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                      // STVK
                      const unsigned int comp_j =
                        fe_primal.system_to_component_index(j).first;
                      if (comp_j == 0 || comp_j == 1)
                        {
                          local_matrix(j, i) +=
                            (scalar_product(
                              piola_kirchhoff_stress_structure_STVK_LinALL,
                              phi_i_grads_v[j])) *
                            fe_values.JxW(q);
                        }
                      else if (comp_j == 2 || comp_j == 3)
                        {
                          local_matrix(j, i) += density_structure * phi_i_v[i] *
                                                phi_i_u[j] * fe_values.JxW(q);
                        }
                      else if (comp_j == 4)
                        {
                          // Artificial Laplace continuation of fluid pressure
                          // into the solid in order to avoid singular system
                          // matrix. An alternative would be to use the
                          // FE_Nothing element.
                          local_matrix(j, i) +=
                            alpha_u *
                            (phi_i_grads_p[i] * phi_i_grads_p[j] +
                             phi_i_p[i] * phi_i_p[j]) *
                            fe_values.JxW(q);
                        }
                      // end j dofs
                    }
                  // end i dofs
                }
              // end n_q_points
            }


          cell->get_dof_indices(local_dof_indices);
          constraints_primal.distribute_local_to_global(local_matrix,
                                                        local_dof_indices,
                                                        system_matrix_primal);
          // end if (second PDE: STVK material)
        }
      // end cell
    }

  timer.exit_section();
}



// In this function we assemble the semi-linear
// of the right hand side of Newton's method (its residual).
// The framework is in principal the same as for the
// system matrix.
template <int dim>
void
FSI_PU_DWR_Problem<dim>::assemble_rhs_primal()
{
  timer.enter_section("Assemble primal rhs.");
  system_rhs_primal = 0;

  QGauss<dim>     quadrature_formula(degree + 2);
  QGauss<dim - 1> face_quadrature_formula(degree + 2);

  FEValues<dim> fe_values(fe_primal,
                          quadrature_formula,
                          update_values | update_quadrature_points |
                            update_JxW_values | update_gradients);

  FEFaceValues<dim> fe_face_values(fe_primal,
                                   face_quadrature_formula,
                                   update_values | update_quadrature_points |
                                     update_normal_vectors | update_gradients |
                                     update_JxW_values);

  const unsigned int dofs_per_cell = fe_primal.dofs_per_cell;

  const unsigned int n_q_points      = quadrature_formula.size();
  const unsigned int n_face_q_points = face_quadrature_formula.size();

  Vector<double> local_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  const FEValuesExtractors::Vector velocities(0);
  const FEValuesExtractors::Vector displacements(dim);
  const FEValuesExtractors::Scalar pressure(dim + dim);

  std::vector<Vector<double>> old_solution_values(n_q_points,
                                                  Vector<double>(dim + dim +
                                                                 1));

  std::vector<std::vector<Tensor<1, dim>>> old_solution_grads(
    n_q_points, std::vector<Tensor<1, dim>>(dim + dim + 1));


  std::vector<Vector<double>> old_solution_face_values(n_face_q_points,
                                                       Vector<double>(dim +
                                                                      dim + 1));

  std::vector<std::vector<Tensor<1, dim>>> old_solution_face_grads(
    n_face_q_points, std::vector<Tensor<1, dim>>(dim + dim + 1));



  typename DoFHandler<dim>::active_cell_iterator cell = dof_handler_primal
                                                          .begin_active(),
                                                 endc =
                                                   dof_handler_primal.end();

  for (; cell != endc; ++cell)
    {
      fe_values.reinit(cell);
      local_rhs = 0;

      cell_diameter = cell->diameter();

      // old Newton iteration
      fe_values.get_function_values(solution_primal, old_solution_values);
      fe_values.get_function_gradients(solution_primal, old_solution_grads);


      // Again, material_id == 0 corresponds to
      // the domain for fluid equations
      if (cell->material_id() == 0)
        {
          for (unsigned int q = 0; q < n_q_points; ++q)
            {
              const Tensor<2, dim> pI =
                ALE_Transformations ::get_pI<dim>(q, old_solution_values);

              const Tensor<1, dim> v =
                ALE_Transformations ::get_v<dim>(q, old_solution_values);

              const Tensor<2, dim> grad_v =
                ALE_Transformations ::get_grad_v<dim>(q, old_solution_grads);

              const Tensor<2, dim> grad_u =
                ALE_Transformations ::get_grad_u<dim>(q, old_solution_grads);

              const Tensor<2, dim> grad_v_T =
                ALE_Transformations ::get_grad_v_T<dim>(grad_v);

              const Tensor<2, dim> F =
                ALE_Transformations ::get_F<dim>(q, old_solution_grads);

              const Tensor<2, dim> F_Inverse =
                ALE_Transformations ::get_F_Inverse<dim>(F);

              const Tensor<2, dim> F_Inverse_T =
                ALE_Transformations ::get_F_Inverse_T<dim>(F_Inverse);

              const double J = ALE_Transformations ::get_J<dim>(F);


              // This is the fluid stress tensor in ALE formulation
              const Tensor<2, dim> sigma_ALE =
                NSE_in_ALE ::get_stress_fluid_except_pressure_ALE<dim>(
                  density_fluid,
                  viscosity,
                  grad_v,
                  grad_v_T,
                  F_Inverse,
                  F_Inverse_T);


              Tensor<2, dim> stress_fluid;
              stress_fluid.clear();
              stress_fluid = (J * sigma_ALE * F_Inverse_T);

              Tensor<2, dim> fluid_pressure;
              fluid_pressure.clear();
              fluid_pressure = (-pI * J * F_Inverse_T);


              // Divergence of the fluid in the ALE formulation
              const double incompressiblity_fluid =
                NSE_in_ALE ::get_Incompressibility_ALE<dim>(q,
                                                            old_solution_grads);

              // Convection term of the fluid in the ALE formulation.
              // We emphasize that the fluid convection term for
              // non-stationary flow problems in ALE
              // representation is difficult to derive.
              // For adequate discretization, the convection term will be
              // split into three smaller terms:
              Tensor<1, dim> convection_fluid;
              convection_fluid.clear();
              convection_fluid = density_fluid * J * (grad_v * F_Inverse * v);

              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  // Fluid, NSE in ALE
                  const unsigned int comp_i =
                    fe_primal.system_to_component_index(i).first;
                  if (comp_i == 0 || comp_i == 1)
                    {
                      const Tensor<1, dim> phi_i_v =
                        fe_values[velocities].value(i, q);
                      const Tensor<2, dim> phi_i_grads_v =
                        fe_values[velocities].gradient(i, q);

                      local_rhs(i) -=
                        (convection_fluid * phi_i_v +
                         scalar_product(fluid_pressure, phi_i_grads_v) +
                         scalar_product(stress_fluid, phi_i_grads_v)) *
                        fe_values.JxW(q);
                    }
                  else if (comp_i == 2 || comp_i == 3)
                    {
                      const Tensor<2, dim> phi_i_grads_u =
                        fe_values[displacements].gradient(i, q);

                      // Linear harmonic MMPDE
                      local_rhs(i) -=
                        (alpha_u * scalar_product(grad_u, phi_i_grads_u)) *
                        fe_values.JxW(q);
                    }
                  else if (comp_i == 4)
                    {
                      const double phi_i_p = fe_values[pressure].value(i, q);
                      local_rhs(i) -=
                        (incompressiblity_fluid * phi_i_p) * fe_values.JxW(q);
                    }
                  // end i dofs
                }
              // close n_q_points
            }

          // As already discussed in the assembling method for the matrix,
          // we have to integrate some terms on the outflow boundary:
          for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
               ++face)
            {
              if (cell->face(face)->at_boundary() &&
                  (cell->face(face)->boundary_id() == 1))
                {
                  fe_face_values.reinit(cell, face);

                  fe_face_values.get_function_values(solution_primal,
                                                     old_solution_face_values);
                  fe_face_values.get_function_gradients(
                    solution_primal, old_solution_face_grads);


                  for (unsigned int q = 0; q < n_face_q_points; ++q)
                    {
                      // These are terms coming from the
                      // previous Newton iterations ...
                      const Tensor<2, dim> grad_v =
                        ALE_Transformations ::get_grad_v<dim>(
                          q, old_solution_face_grads);

                      const Tensor<2, dim> grad_v_T =
                        ALE_Transformations ::get_grad_v_T<dim>(grad_v);

                      const Tensor<2, dim> F = ALE_Transformations ::get_F<dim>(
                        q, old_solution_face_grads);

                      const Tensor<2, dim> F_Inverse =
                        ALE_Transformations ::get_F_Inverse<dim>(F);

                      const Tensor<2, dim> F_Inverse_T =
                        ALE_Transformations ::get_F_Inverse_T<dim>(F_Inverse);

                      const double J = ALE_Transformations ::get_J<dim>(F);


                      Tensor<2, dim> sigma_ALE_tilde;
                      sigma_ALE_tilde.clear();
                      sigma_ALE_tilde =
                        (density_fluid * viscosity * F_Inverse_T * grad_v_T);

                      // Neumann boundary integral
                      Tensor<2, dim> stress_fluid_transposed_part;
                      stress_fluid_transposed_part.clear();
                      stress_fluid_transposed_part =
                        (J * sigma_ALE_tilde * F_Inverse_T);

                      const Tensor<1, dim> neumann_value =
                        (stress_fluid_transposed_part *
                         fe_face_values.normal_vector(q));


                      for (unsigned int i = 0; i < dofs_per_cell; ++i)
                        {
                          const unsigned int comp_i =
                            fe_primal.system_to_component_index(i).first;
                          if (comp_i == 0 || comp_i == 1)
                            {
                              local_rhs(i) +=
                                neumann_value *
                                fe_face_values[velocities].value(i, q) *
                                fe_face_values.JxW(q);
                            }
                          // end i
                        }
                      // end face_n_q_points
                    }
                }
            } // end face integrals do-nothing condition


          cell->get_dof_indices(local_dof_indices);
          constraints_primal.distribute_local_to_global(local_rhs,
                                                        local_dof_indices,
                                                        system_rhs_primal);

          // Finally, we arrive at the end for assembling
          // the variational formulation for the fluid part and step to
          // the assembling process of the structure terms:
        }
      else if (cell->material_id() == 1)
        {
          for (unsigned int q = 0; q < n_q_points; ++q)
            {
              const Tensor<1, dim> grad_p =
                ALE_Transformations ::get_grad_p<dim>(q, old_solution_grads);

              const double p = old_solution_values[q](dim + dim);

              const Tensor<1, dim> v =
                ALE_Transformations ::get_v<dim>(q, old_solution_values);

              const Tensor<2, dim> F =
                ALE_Transformations ::get_F<dim>(q, old_solution_grads);

              const Tensor<2, dim> F_T = ALE_Transformations ::get_F_T<dim>(F);

              const Tensor<2, dim> Identity =
                ALE_Transformations ::get_Identity<dim>();

              const Tensor<2, dim> F_Inverse =
                ALE_Transformations ::get_F_Inverse<dim>(F);

              const Tensor<2, dim> F_Inverse_T =
                ALE_Transformations ::get_F_Inverse_T<dim>(F_Inverse);

              const double J = ALE_Transformations ::get_J<dim>(F);

              const Tensor<2, dim> E =
                Structure_Terms_in_ALE ::get_E<dim>(F_T, F, Identity);

              const double tr_E = Structure_Terms_in_ALE ::get_tr_E<dim>(E);



              // STVK structure model
              Tensor<2, dim> sigma_structure_ALE;
              sigma_structure_ALE.clear();
              sigma_structure_ALE =
                (1.0 / J * F *
                 (lame_coefficient_lambda * tr_E * Identity +
                  2 * lame_coefficient_mu * E) *
                 F_T);


              Tensor<2, dim> stress_term;
              stress_term.clear();
              stress_term = (J * sigma_structure_ALE * F_Inverse_T);

              // Attention: normally no time
              Tensor<1, dim> structure_force;
              structure_force.clear();
              structure_force[0] = density_structure * force_structure_x;
              structure_force[1] = density_structure * force_structure_y;

              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  // STVK structure model
                  const unsigned int comp_i =
                    fe_primal.system_to_component_index(i).first;
                  if (comp_i == 0 || comp_i == 1)
                    {
                      const Tensor<1, dim> phi_i_v =
                        fe_values[velocities].value(i, q);
                      const Tensor<2, dim> phi_i_grads_v =
                        fe_values[velocities].gradient(i, q);

                      local_rhs(i) -=
                        (scalar_product(stress_term, phi_i_grads_v) +
                         structure_force * phi_i_v) *
                        fe_values.JxW(q);
                    }
                  else if (comp_i == 2 || comp_i == 3)
                    {
                      const Tensor<1, dim> phi_i_u =
                        fe_values[displacements].value(i, q);
                      local_rhs(i) -=
                        density_structure * v * phi_i_u * fe_values.JxW(q);
                    }
                  else if (comp_i == 4)
                    {
                      // Laplace continuation pressure. Explanation see in the
                      // assemble_matrix part.
                      const Tensor<1, dim> phi_i_grads_p =
                        fe_values[pressure].gradient(i, q);
                      const double phi_i_p = fe_values[pressure].value(i, q);
                      local_rhs(i) -= alpha_u *
                                      (grad_p * phi_i_grads_p + p * phi_i_p) *
                                      fe_values.JxW(q);
                    }
                  // end i
                }
              // end n_q_points
            }

          cell->get_dof_indices(local_dof_indices);
          constraints_primal.distribute_local_to_global(local_rhs,
                                                        local_dof_indices,
                                                        system_rhs_primal);

          // end if (for STVK material)
        }

    } // end cell

  timer.exit_section();
}


// Here, we impose boundary conditions
// for the whole system. The fluid inflow
// is prescribed by a parabolic profile. The usual
// structure displacement shall be fixed
// at all outer boundaries.
// The pressure variable is not subjected to any
// Dirichlet boundary conditions and is left free
// in this method. Please note, that
// the interface between fluid and structure has no
// physical boundary due to our formulation. Interface
// conditions are automatically fulfilled: that is
// one major advantage of the `variational-monolithic' formulation.
template <int dim>
void
FSI_PU_DWR_Problem<dim>::set_initial_bc_primal()
{
  // Only place holder since no time-dependent problem is considered here.
  double time = 0.0;

  std::map<types::global_dof_index, double> boundary_values;
  std::vector<bool>                         component_mask(dim + dim + 1, true);
  // (Scalar) pressure
  component_mask[dim + dim] = false;

  VectorTools::interpolate_boundary_values(dof_handler_primal,
                                           0,
                                           BoundaryParabel<dim>(time),
                                           boundary_values,
                                           component_mask);


  component_mask[dim] = false; // ux
  VectorTools::interpolate_boundary_values(dof_handler_primal,
                                           2,
                                           ZeroFunction<dim>(dim + dim + 1),
                                           boundary_values,
                                           component_mask);


  VectorTools::interpolate_boundary_values(dof_handler_primal,
                                           3,
                                           ZeroFunction<dim>(dim + dim + 1),
                                           boundary_values,
                                           component_mask);


  component_mask[dim] = true; // ux
  VectorTools::interpolate_boundary_values(dof_handler_primal,
                                           80,
                                           ZeroFunction<dim>(dim + dim + 1),
                                           boundary_values,
                                           component_mask);

  VectorTools::interpolate_boundary_values(dof_handler_primal,
                                           81,
                                           ZeroFunction<dim>(dim + dim + 1),
                                           boundary_values,
                                           component_mask);

  component_mask[0] = false;
  component_mask[1] = false;

  VectorTools::interpolate_boundary_values(dof_handler_primal,
                                           1,
                                           ZeroFunction<dim>(dim + dim + 1),
                                           boundary_values,
                                           component_mask);

  for (typename std::map<types::global_dof_index, double>::const_iterator i =
         boundary_values.begin();
       i != boundary_values.end();
       ++i)
    solution_primal(i->first) = i->second;
}

// This function applies boundary conditions
// to the Newton iteration steps. For all variables that
// have Dirichlet conditions on some (or all) parts
// of the outer boundary, we apply zero-Dirichlet
// conditions, now.
template <int dim>
void
FSI_PU_DWR_Problem<dim>::set_newton_bc_primal()
{
  std::vector<bool> component_mask(dim + dim + 1, true);
  component_mask[dim + dim] = false; // p


  VectorTools::interpolate_boundary_values(dof_handler_primal,
                                           0,
                                           ZeroFunction<dim>(dim + dim + 1),
                                           constraints_primal,
                                           component_mask);
  component_mask[dim] = false; // ux
  VectorTools::interpolate_boundary_values(dof_handler_primal,
                                           2,
                                           ZeroFunction<dim>(dim + dim + 1),
                                           constraints_primal,
                                           component_mask);

  VectorTools::interpolate_boundary_values(dof_handler_primal,
                                           3,
                                           ZeroFunction<dim>(dim + dim + 1),
                                           constraints_primal,
                                           component_mask);
  component_mask[dim] = true; // ux
  VectorTools::interpolate_boundary_values(dof_handler_primal,
                                           80,
                                           ZeroFunction<dim>(dim + dim + 1),
                                           constraints_primal,
                                           component_mask);
  VectorTools::interpolate_boundary_values(dof_handler_primal,
                                           81,
                                           ZeroFunction<dim>(dim + dim + 1),
                                           constraints_primal,
                                           component_mask);
  component_mask[0] = false;
  component_mask[1] = false;

  VectorTools::interpolate_boundary_values(dof_handler_primal,
                                           1,
                                           ZeroFunction<dim>(dim + dim + 1),
                                           constraints_primal,
                                           component_mask);
}

// In this function, we solve the linear systems
// inside the nonlinear Newton iteration. For simplicity we
// use a direct solver from UMFPACK.
template <int dim>
void
FSI_PU_DWR_Problem<dim>::solve_primal()
{
  timer.enter_section("Solve primal linear system.");
  Vector<double> sol, rhs;
  sol = newton_update_primal;
  rhs = system_rhs_primal;

  A_direct_primal.vmult(sol, rhs);
  newton_update_primal = sol;

  constraints_primal.distribute(newton_update_primal);
  timer.exit_section();
}

// This is the Newton iteration with simple linesearch backtracking
// to solve the
// non-linear system of equations. First, we declare some
// standard parameters of the solution method. Addionally,
// we also implement an easy line search algorithm.
template <int dim>
void
FSI_PU_DWR_Problem<dim>::newton_iteration_primal()

{
  Timer timer_newton;
  Timer timer_newton_global;
  timer_newton_global.start();

  const unsigned int max_no_newton_steps = 60;

  // Decision whether the system matrix should be build
  // at each Newton step
  const double nonlinear_rho = 0.1;

  // Line search parameters
  unsigned int       line_search_step;
  const unsigned int max_no_line_search_steps = 10;
  const double       line_search_damping      = 0.6;
  double             new_newton_residual;

  // Application of the initial boundary conditions to the
  // variational equations:
  set_initial_bc_primal();
  assemble_rhs_primal();

  double       newton_residual     = system_rhs_primal.linfty_norm();
  double       initial_residual    = newton_residual;
  double       old_newton_residual = newton_residual;
  double       solution_norm       = 1.0;
  double       old_solution_norm   = 2.0;
  unsigned int newton_step         = 1;

  // Output explanation
  std::cout << "Iter" << '\t' << "Res (abs)" << '\t' << "Res (rel)" << '\t'
            << "Iter-Err (abs)" << '\t' << "Iter-Err (rel)" << '\t'
            << "Res reduct" << '\t' << "Reb Jac" << '\t' << "LS" << '\t'
            << "Wall time" << std::endl;

  if (newton_residual < lower_bound_newton_residual)
    {
      std::cout << '\t' << std::scientific << newton_residual << std::endl;
    }

  while (newton_residual > lower_bound_newton_residual &&
         newton_residual / initial_residual >
           lower_bound_newton_residual_relative &&
         newton_step < max_no_newton_steps)
    {
      timer_newton.start();
      old_newton_residual = newton_residual;
      old_solution_norm   = solution_primal.linfty_norm();

      assemble_rhs_primal();
      newton_residual = system_rhs_primal.linfty_norm();

      if (newton_residual < lower_bound_newton_residual ||
          (newton_residual / initial_residual <
           lower_bound_newton_residual_relative))
        {
          std::cout << '\t' << std::scientific << newton_residual << '\t'
                    << std::scientific << newton_residual / initial_residual
                    << '\t' << std::endl;
          break;
        }

      if (newton_residual / old_newton_residual > nonlinear_rho)
        {
          assemble_matrix_primal();
          // Only factorize when matrix is re-built
          A_direct_primal.factorize(system_matrix_primal);
        }

      // Solve Ax = b
      solve_primal();

      line_search_step = 0;
      for (; line_search_step < max_no_line_search_steps; ++line_search_step)
        {
          solution_primal += newton_update_primal;

          assemble_rhs_primal();
          new_newton_residual = system_rhs_primal.linfty_norm();

          if (new_newton_residual < newton_residual)
            break;
          else
            solution_primal -= newton_update_primal;

          newton_update_primal *= line_search_damping;
        }

      timer_newton.stop();

      solution_norm = solution_primal.linfty_norm();

      std::cout << std::setprecision(5) << newton_step << '\t'
                << std::scientific << newton_residual << '\t' << std::scientific
                << newton_residual / initial_residual << '\t' << std::scientific
                << std::abs(solution_norm - old_solution_norm) << '\t'
                << std::scientific
                << std::abs(solution_norm - old_solution_norm) /
                     old_solution_norm
                << '\t' << std::scientific
                << newton_residual / old_newton_residual << '\t';
      if (newton_residual / old_newton_residual > nonlinear_rho)
        std::cout << "r" << '\t';
      else
        std::cout << " " << '\t';
      std::cout << line_search_step << '\t' << std::scientific
                << timer_newton.cpu_time() << std::endl;


      // Updates
      timer_newton.reset();
      newton_step++;
    }

  timer_newton_global.stop();
  std::cout << "Wall time solving primal system:  " << timer_newton_global()
            << std::endl;
  timer_newton_global.reset();
}



////////////////////////////////////////////////////////////////////////////////////////////
// Adjoint problem implementation.


// This function is similar to many deal.II tuturial steps.
template <int dim>
void
FSI_PU_DWR_Problem<dim>::setup_system_adjoint()
{
  timer.enter_section("Setup adjoint system.");

  system_matrix_adjoint.clear();

  dof_handler_adjoint.distribute_dofs(fe_adjoint);
  DoFRenumbering::Cuthill_McKee(dof_handler_adjoint);

  // We are dealing with 7 components for this
  // two-dimensional fluid-structure interacion problem
  // Precisely, we use:
  // velocity in x and y:                0
  // structure displacement in x and y:  1
  // scalar pressure field:              2
  std::vector<unsigned int> block_component(5, 0);
  block_component[dim]       = 1;
  block_component[dim + 1]   = 1;
  block_component[dim + dim] = 2;

  DoFRenumbering::component_wise(dof_handler_adjoint, block_component);

  {
    constraints_adjoint.clear();
    DoFTools::make_hanging_node_constraints(dof_handler_adjoint,
                                            constraints_adjoint);
    set_bc_adjoint();
  }
  constraints_adjoint.close();

  std::vector<types::global_dof_index> dofs_per_block(3);
  DoFTools::count_dofs_per_block(dof_handler_adjoint,
                                 dofs_per_block,
                                 block_component);
  const unsigned int n_v = dofs_per_block[0], n_u = dofs_per_block[1],
                     n_p = dofs_per_block[2];

  std::cout << "DoFs (adjoint):\t" << dof_handler_adjoint.n_dofs() << " ("
            << n_v << '+' << n_u << '+' << n_p << ')' << std::endl;



  {
    BlockDynamicSparsityPattern csp(3, 3);

    csp.block(0, 0).reinit(n_v, n_v);
    csp.block(0, 1).reinit(n_v, n_u);
    csp.block(0, 2).reinit(n_v, n_p);

    csp.block(1, 0).reinit(n_u, n_v);
    csp.block(1, 1).reinit(n_u, n_u);
    csp.block(1, 2).reinit(n_u, n_p);

    csp.block(2, 0).reinit(n_p, n_v);
    csp.block(2, 1).reinit(n_p, n_u);
    csp.block(2, 2).reinit(n_p, n_p);

    csp.collect_sizes();


    DoFTools::make_sparsity_pattern(dof_handler_adjoint,
                                    csp,
                                    constraints_adjoint,
                                    false);

    sparsity_pattern_adjoint.copy_from(csp);
  }

  system_matrix_adjoint.reinit(sparsity_pattern_adjoint);

  // Current solution
  solution_adjoint.reinit(3);
  solution_adjoint.block(0).reinit(n_v);
  solution_adjoint.block(1).reinit(n_u);
  solution_adjoint.block(2).reinit(n_p);

  solution_adjoint.collect_sizes();


  // Residual for  Newton's method
  system_rhs_adjoint.reinit(3);
  system_rhs_adjoint.block(0).reinit(n_v);
  system_rhs_adjoint.block(1).reinit(n_u);
  system_rhs_adjoint.block(2).reinit(n_p);

  system_rhs_adjoint.collect_sizes();

  timer.exit_section();
}



template <int dim>
void
FSI_PU_DWR_Problem<dim>::assemble_matrix_adjoint()
{
  timer.enter_section("Assemble adjoint matrix.");
  system_matrix_adjoint = 0;

  // Choose quadrature rule sufficiently high with respect
  // to the finite element choice.
  QGauss<dim>     quadrature_formula(5);
  QGauss<dim - 1> face_quadrature_formula(5);

  FEValues<dim> fe_values(fe_adjoint,
                          quadrature_formula,
                          update_values | update_quadrature_points |
                            update_JxW_values | update_gradients);

  FEFaceValues<dim> fe_face_values(fe_adjoint,
                                   face_quadrature_formula,
                                   update_values | update_quadrature_points |
                                     update_normal_vectors | update_gradients |
                                     update_JxW_values);


  FEValues<dim> fe_values_primal(fe_primal,
                                 quadrature_formula,
                                 update_values | update_quadrature_points |
                                   update_JxW_values | update_gradients);

  FEFaceValues<dim> fe_face_values_primal(fe_primal,
                                          face_quadrature_formula,
                                          update_values |
                                            update_quadrature_points |
                                            update_normal_vectors |
                                            update_gradients |
                                            update_JxW_values);


  const unsigned int dofs_per_cell = fe_adjoint.dofs_per_cell;

  const unsigned int n_q_points      = quadrature_formula.size();
  const unsigned int n_face_q_points = face_quadrature_formula.size();

  FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);


  // Now, we are going to use the
  // FEValuesExtractors to determine
  // the four principle variables
  const FEValuesExtractors::Vector velocities(0);       // 0
  const FEValuesExtractors::Vector displacements(dim);  // 2
  const FEValuesExtractors::Scalar pressure(dim + dim); // 4

  // We declare Vectors and Tensors for
  // the solutions at the previous Newton iteration:
  std::vector<Vector<double>> old_solution_values(n_q_points,
                                                  Vector<double>(dim + dim +
                                                                 1));

  std::vector<std::vector<Tensor<1, dim>>> old_solution_grads(
    n_q_points, std::vector<Tensor<1, dim>>(dim + dim + 1));

  std::vector<Vector<double>> old_solution_face_values(n_face_q_points,
                                                       Vector<double>(dim +
                                                                      dim + 1));

  std::vector<std::vector<Tensor<1, dim>>> old_solution_face_grads(
    n_face_q_points, std::vector<Tensor<1, dim>>(dim + dim + 1));



  // Primal solution values
  std::vector<Vector<double>> old_solution_values_primal(
    n_q_points, Vector<double>(dim + dim + 1));

  std::vector<std::vector<Tensor<1, dim>>> old_solution_grads_primal(
    n_q_points, std::vector<Tensor<1, dim>>(dim + dim + 1));


  std::vector<Vector<double>> old_solution_face_values_primal(
    n_face_q_points, Vector<double>(dim + dim + 1));

  std::vector<std::vector<Tensor<1, dim>>> old_solution_face_grads_primal(
    n_face_q_points, std::vector<Tensor<1, dim>>(dim + dim + 1));



  // Declaring test functions:
  std::vector<Tensor<1, dim>> phi_i_v(dofs_per_cell);
  std::vector<Tensor<2, dim>> phi_i_grads_v(dofs_per_cell);
  std::vector<double>         phi_i_p(dofs_per_cell);
  std::vector<Tensor<1, dim>> phi_i_grads_p(dofs_per_cell);
  std::vector<Tensor<1, dim>> phi_i_u(dofs_per_cell);
  std::vector<Tensor<2, dim>> phi_i_grads_u(dofs_per_cell);

  // This is the identity matrix in two dimensions:
  const Tensor<2, dim> Identity = ALE_Transformations ::get_Identity<dim>();

  typename DoFHandler<dim>::active_cell_iterator cell = dof_handler_adjoint
                                                          .begin_active(),
                                                 endc =
                                                   dof_handler_adjoint.end();

  typename DoFHandler<dim>::active_cell_iterator cell_primal =
    dof_handler_primal.begin_active();

  for (; cell != endc; ++cell, ++cell_primal)
    {
      fe_values.reinit(cell);
      fe_values_primal.reinit(cell_primal);

      local_matrix = 0;

      // We need the cell diameter to control the fluid mesh motion
      cell_diameter = cell->diameter();

      // Old Newton iteration values
      fe_values.get_function_values(solution_adjoint, old_solution_values);
      fe_values.get_function_gradients(solution_adjoint, old_solution_grads);

      // Old primal Newton iteration values
      fe_values_primal.get_function_values(solution_primal,
                                           old_solution_values_primal);
      fe_values_primal.get_function_gradients(solution_primal,
                                              old_solution_grads_primal);


      // Next, we run over all cells for the fluid equations
      if (cell->material_id() == 0)
        {
          for (unsigned int q = 0; q < n_q_points; ++q)
            {
              for (unsigned int k = 0; k < dofs_per_cell; ++k)
                {
                  phi_i_v[k]       = fe_values[velocities].value(k, q);
                  phi_i_grads_v[k] = fe_values[velocities].gradient(k, q);
                  phi_i_p[k]       = fe_values[pressure].value(k, q);
                  phi_i_grads_p[k] = fe_values[pressure].gradient(k, q);
                  phi_i_u[k]       = fe_values[displacements].value(k, q);
                  phi_i_grads_u[k] = fe_values[displacements].gradient(k, q);
                }

              // All `Newton values' come from the primal solution.

              const Tensor<2, dim> pI =
                ALE_Transformations ::get_pI<dim>(q,
                                                  old_solution_values_primal);

              const Tensor<1, dim> v =
                ALE_Transformations ::get_v<dim>(q, old_solution_values_primal);

              const Tensor<2, dim> grad_v =
                ALE_Transformations ::get_grad_v<dim>(
                  q, old_solution_grads_primal);

              const Tensor<2, dim> grad_v_T =
                ALE_Transformations ::get_grad_v_T<dim>(grad_v);

              const Tensor<2, dim> F =
                ALE_Transformations ::get_F<dim>(q, old_solution_grads_primal);

              const Tensor<2, dim> F_Inverse =
                ALE_Transformations ::get_F_Inverse<dim>(F);

              const Tensor<2, dim> F_Inverse_T =
                ALE_Transformations ::get_F_Inverse_T<dim>(F_Inverse);

              const double J = ALE_Transformations ::get_J<dim>(F);

              // non-transformed in Eulerian coordinates.
              // Needs to be transformed for FSI
              // Tensor<1,2> convection_fluid = grad_v * v;


              // Stress tensor for the fluid in ALE notation
              const Tensor<2, dim> sigma_ALE =
                NSE_in_ALE ::get_stress_fluid_ALE<dim>(density_fluid,
                                                       viscosity,
                                                       pI,
                                                       grad_v,
                                                       grad_v_T,
                                                       F_Inverse,
                                                       F_Inverse_T);

              // Outer loop for dofs
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  const Tensor<2, dim> pI_LinP =
                    ALE_Transformations ::get_pI_LinP<dim>(phi_i_p[i]);

                  const Tensor<2, dim> grad_v_LinV =
                    ALE_Transformations ::get_grad_v_LinV<dim>(
                      phi_i_grads_v[i]);

                  const double J_LinU = ALE_Transformations ::get_J_LinU<dim>(
                    q, old_solution_grads_primal, phi_i_grads_u[i]);

                  const Tensor<2, dim> J_F_Inverse_T_LinU =
                    ALE_Transformations ::get_J_F_Inverse_T_LinU<dim>(
                      phi_i_grads_u[i]);

                  const Tensor<2, dim> F_Inverse_LinU =
                    ALE_Transformations ::get_F_Inverse_LinU(
                      phi_i_grads_u[i],
                      J,
                      J_LinU,
                      q,
                      old_solution_grads_primal);

                  const Tensor<2, dim> stress_fluid_ALE_1st_term_LinAll =
                    NSE_in_ALE ::get_stress_fluid_ALE_1st_term_LinAll<dim>(
                      pI, F_Inverse_T, J_F_Inverse_T_LinU, pI_LinP, J);

                  const Tensor<2, dim> stress_fluid_ALE_2nd_term_LinAll =
                    NSE_in_ALE ::get_stress_fluid_ALE_2nd_term_LinAll_short(
                      J_F_Inverse_T_LinU,
                      sigma_ALE,
                      grad_v,
                      grad_v_LinV,
                      F_Inverse,
                      F_Inverse_LinU,
                      J,
                      viscosity,
                      density_fluid);

                  const Tensor<1, dim> convection_fluid_LinAll_short =
                    NSE_in_ALE ::get_Convection_LinAll_short<dim>(
                      phi_i_grads_v[i],
                      phi_i_v[i],
                      J,
                      J_LinU,
                      F_Inverse,
                      F_Inverse_LinU,
                      v,
                      grad_v,
                      density_fluid);

                  const double incompressibility_ALE_LinAll =
                    NSE_in_ALE ::get_Incompressibility_ALE_LinAll<dim>(
                      phi_i_grads_v[i],
                      phi_i_grads_u[i],
                      q,
                      old_solution_grads_primal);



                  // Inner loop for dofs
                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                      // Info: In the adjoint matrix, the entries are flipped,
                      // i.e., (i,j) (and not (j,i)) because the adjoint matrix
                      // is transposed by definition.

                      // Fluid , NSE in ALE adjoint
                      const unsigned int comp_j =
                        fe_adjoint.system_to_component_index(j).first;
                      if (comp_j == 0 || comp_j == 1)
                        {
                          local_matrix(i, j) +=
                            (convection_fluid_LinAll_short * phi_i_v[j] +
                             scalar_product(stress_fluid_ALE_1st_term_LinAll,
                                            phi_i_grads_v[j]) +
                             scalar_product(stress_fluid_ALE_2nd_term_LinAll,
                                            phi_i_grads_v[j])) *
                            fe_values.JxW(q);
                        }
                      else if (comp_j == 2 || comp_j == 3)
                        {
                          // Linear harmonic MMPDE
                          local_matrix(i, j) +=
                            (alpha_u * scalar_product(phi_i_grads_u[i],
                                                      phi_i_grads_u[j])) *
                            fe_values.JxW(q);
                        }
                      else if (comp_j == 4)
                        {
                          local_matrix(i, j) +=
                            (incompressibility_ALE_LinAll * phi_i_p[j]) *
                            fe_values.JxW(q);
                        }

                    } // end j dofs

                } // end i dofs

            } // end n_q_points

          // do nothing  outflow condition in adjoint form

          for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
               ++face)
            {
              if (cell->face(face)->at_boundary() &&
                  (cell->face(face)->boundary_id() == 1))
                {
                  fe_face_values.reinit(cell, face);
                  fe_face_values_primal.reinit(cell_primal, face);

                  fe_face_values.get_function_values(solution_adjoint,
                                                     old_solution_face_values);
                  fe_face_values.get_function_gradients(
                    solution_adjoint, old_solution_face_grads);

                  // Old primal Newton iteration values
                  fe_face_values_primal.get_function_values(
                    solution_primal, old_solution_face_values_primal);
                  fe_face_values_primal.get_function_gradients(
                    solution_primal, old_solution_face_grads_primal);

                  for (unsigned int q = 0; q < n_face_q_points; ++q)
                    {
                      for (unsigned int k = 0; k < dofs_per_cell; ++k)
                        {
                          phi_i_v[k] = fe_face_values[velocities].value(k, q);
                          phi_i_grads_v[k] =
                            fe_face_values[velocities].gradient(k, q);
                          phi_i_grads_u[k] =
                            fe_face_values[displacements].gradient(k, q);
                        }

                      const Tensor<2, dim> grad_v =
                        ALE_Transformations ::get_grad_v<dim>(
                          q, old_solution_face_grads_primal);

                      const Tensor<2, dim> F = ALE_Transformations ::get_F<dim>(
                        q, old_solution_face_grads_primal);

                      const Tensor<2, dim> F_Inverse =
                        ALE_Transformations ::get_F_Inverse<dim>(F);

                      const double J = ALE_Transformations ::get_J<dim>(F);


                      for (unsigned int i = 0; i < dofs_per_cell; ++i)
                        {
                          const Tensor<2, dim> grad_v_LinV =
                            ALE_Transformations ::get_grad_v_LinV<dim>(
                              phi_i_grads_v[i]);

                          const double J_LinU =
                            ALE_Transformations ::get_J_LinU<dim>(
                              q,
                              old_solution_face_grads_primal,
                              phi_i_grads_u[i]);

                          const Tensor<2, dim> J_F_Inverse_T_LinU =
                            ALE_Transformations ::get_J_F_Inverse_T_LinU<dim>(
                              phi_i_grads_u[i]);

                          const Tensor<2, dim> F_Inverse_LinU =
                            ALE_Transformations ::get_F_Inverse_LinU(
                              phi_i_grads_u[i],
                              J,
                              J_LinU,
                              q,
                              old_solution_face_grads_primal);

                          const Tensor<2, dim>
                            stress_fluid_ALE_3rd_term_LinAll = NSE_in_ALE ::
                              get_stress_fluid_ALE_3rd_term_LinAll_short<dim>(
                                F_Inverse,
                                F_Inverse_LinU,
                                grad_v,
                                grad_v_LinV,
                                viscosity,
                                density_fluid,
                                J,
                                J_F_Inverse_T_LinU);

                          // Here, we multiply the symmetric part of fluid's
                          // stress tensor with the normal direction.
                          const Tensor<1, dim> neumann_value =
                            (stress_fluid_ALE_3rd_term_LinAll *
                             fe_face_values.normal_vector(q));

                          for (unsigned int j = 0; j < dofs_per_cell; ++j)
                            {
                              const unsigned int comp_j =
                                fe_adjoint.system_to_component_index(j).first;
                              if (comp_j == 0 || comp_j == 1)
                                {
                                  // Flip again matrix entries
                                  local_matrix(i, j) -= neumann_value *
                                                        phi_i_v[j] *
                                                        fe_face_values.JxW(q);
                                }
                              // end j
                            }
                          // end i
                        }
                      // end q_face_points
                    }
                  // end if-routine face integrals
                }
              // end face integrals do-nothing
            }



          // This is the same as discussed in step-22:
          cell->get_dof_indices(local_dof_indices);
          constraints_adjoint.distribute_local_to_global(local_matrix,
                                                         local_dof_indices,
                                                         system_matrix_adjoint);

          // Finally, we arrive at the end for assembling the matrix
          // for the fluid equations and step to the computation of the
          // structure terms:
        }
      else if (cell->material_id() == 1)
        {
          for (unsigned int q = 0; q < n_q_points; ++q)
            {
              for (unsigned int k = 0; k < dofs_per_cell; ++k)
                {
                  phi_i_v[k]       = fe_values[velocities].value(k, q);
                  phi_i_grads_v[k] = fe_values[velocities].gradient(k, q);
                  phi_i_p[k]       = fe_values[pressure].value(k, q);
                  phi_i_grads_p[k] = fe_values[pressure].gradient(k, q);
                  phi_i_u[k]       = fe_values[displacements].value(k, q);
                  phi_i_grads_u[k] = fe_values[displacements].gradient(k, q);
                }

              // It is here the same as already shown for the fluid equations.
              // First, we prepare things coming from the previous Newton
              // iteration...
              const Tensor<2, dim> F =
                ALE_Transformations ::get_F<dim>(q, old_solution_grads_primal);

              const Tensor<2, dim> F_T = ALE_Transformations ::get_F_T<dim>(F);


              const Tensor<2, dim> E =
                Structure_Terms_in_ALE ::get_E<dim>(F_T, F, Identity);

              const double tr_E = Structure_Terms_in_ALE ::get_tr_E<dim>(E);


              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  const Tensor<2, dim> F_LinU =
                    ALE_Transformations ::get_F_LinU<dim>(phi_i_grads_u[i]);


                  // STVK: Green-Lagrange strain tensor derivatives
                  const Tensor<2, dim> E_LinU =
                    0.5 * (transpose(F_LinU) * F + transpose(F) * F_LinU);

                  const double tr_E_LinU =
                    Structure_Terms_in_ALE ::get_tr_E_LinU<dim>(
                      q, old_solution_grads, phi_i_grads_u[i]);


                  // STVK
                  // Piola-kirchhoff stress structure STVK linearized in all
                  // directions
                  Tensor<2, dim> piola_kirchhoff_stress_structure_STVK_LinALL;
                  piola_kirchhoff_stress_structure_STVK_LinALL =
                    lame_coefficient_lambda *
                      (F_LinU * tr_E * Identity + F * tr_E_LinU * Identity) +
                    2 * lame_coefficient_mu * (F_LinU * E + F * E_LinU);


                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                      // STVK
                      const unsigned int comp_j =
                        fe_adjoint.system_to_component_index(j).first;
                      if (comp_j == 0 || comp_j == 1)
                        {
                          local_matrix(i, j) +=
                            (scalar_product(
                              piola_kirchhoff_stress_structure_STVK_LinALL,
                              phi_i_grads_v[j])) *
                            fe_values.JxW(q);
                        }
                      else if (comp_j == 2 || comp_j == 3)
                        {
                          local_matrix(i, j) +=
                            (density_structure * phi_i_v[i] * phi_i_u[j]) *
                            fe_values.JxW(q);
                        }
                      else if (comp_j == 4)
                        {
                          local_matrix(i, j) +=
                            alpha_u * (phi_i_grads_p[i] * phi_i_grads_p[j]) *
                            fe_values.JxW(q);
                        }
                      // end j dofs
                    }
                  // end i dofs
                }
              // end n_q_points
            }


          cell->get_dof_indices(local_dof_indices);
          constraints_adjoint.distribute_local_to_global(local_matrix,
                                                         local_dof_indices,
                                                         system_matrix_adjoint);



        } // end if (second PDE: STVK material)
          // end cell
    }

  timer.exit_section();
}



// Implement goal functional right hand side: drag
template <int dim>
void
FSI_PU_DWR_Problem<dim>::assemble_rhs_adjoint_drag()
{
  timer.enter_section("Assemble adjoint rhs.");
  system_rhs_adjoint = 0;

  // Info: Quadrature degree must be sufficiently high
  QGauss<dim>     quadrature_formula(5);
  QGauss<dim - 1> face_quadrature_formula(5);

  FEValues<dim> fe_values(fe_adjoint,
                          quadrature_formula,
                          update_values | update_quadrature_points |
                            update_JxW_values | update_gradients);


  FEFaceValues<dim> fe_face_values(fe_adjoint,
                                   face_quadrature_formula,
                                   update_values | update_quadrature_points |
                                     update_normal_vectors | update_gradients |
                                     update_JxW_values);

  const unsigned int dofs_per_cell = fe_adjoint.dofs_per_cell;

  // const unsigned int   n_q_points      = quadrature_formula.size();
  const unsigned int n_face_q_points = face_quadrature_formula.size();

  Vector<double> local_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  const FEValuesExtractors::Vector velocities(0);
  const FEValuesExtractors::Vector displacements(dim);
  const FEValuesExtractors::Scalar pressure(dim + dim);

  const Tensor<2, dim> Identity = ALE_Transformations ::get_Identity<dim>();


  typename DoFHandler<dim>::active_cell_iterator cell = dof_handler_adjoint
                                                          .begin_active(),
                                                 endc =
                                                   dof_handler_adjoint.end();


  Tensor<1, 2> drag_lift_value;
  for (; cell != endc; ++cell)
    {
      fe_values.reinit(cell);
      local_rhs = 0;

      // Again, material_id == 0 corresponds to
      // the domain for fluid equations
      if (cell->material_id() == 0)
        {
          // Evaluate drag on the cylinder boundary.
          for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
               ++face)
            {
              if (cell->face(face)->at_boundary() &&
                  (cell->face(face)->boundary_id() == 80))
                {
                  fe_face_values.reinit(cell, face);

                  for (unsigned int q = 0; q < n_face_q_points; ++q)
                    {
                      for (unsigned int i = 0; i < dofs_per_cell; ++i)
                        {
                          const Tensor<2, dim> phi_i_grads_v =
                            fe_face_values[velocities].gradient(i, q);
                          const Tensor<2, dim> phi_i_grads_u =
                            fe_face_values[displacements].gradient(i, q);
                          const double phi_i_p =
                            fe_face_values[pressure].value(i, q);

                          Tensor<2, dim> pI;
                          pI[0][0] = phi_i_p;
                          pI[1][1] = phi_i_p;

                          Tensor<2, dim> grad_v = phi_i_grads_v;

                          const Tensor<2, dim> grad_v_T =
                            ALE_Transformations ::get_grad_v_T<dim>(grad_v);

                          const Tensor<2, dim> F = Identity + phi_i_grads_u;

                          const Tensor<2, dim> F_Inverse =
                            ALE_Transformations ::get_F_Inverse<dim>(F);

                          const Tensor<2, dim> F_Inverse_T =
                            ALE_Transformations ::get_F_Inverse_T<dim>(
                              F_Inverse);

                          const double J = ALE_Transformations ::get_J<dim>(F);

                          const Tensor<2, dim> sigma_ALE =
                            NSE_in_ALE ::get_stress_fluid_except_pressure_ALE<
                              dim>(density_fluid,
                                   viscosity,
                                   grad_v,
                                   grad_v_T,
                                   F_Inverse,
                                   F_Inverse_T);

                          Tensor<2, dim> stress_fluid;
                          stress_fluid.clear();
                          stress_fluid = (J * sigma_ALE * F_Inverse_T);

                          Tensor<2, dim> fluid_pressure;
                          fluid_pressure.clear();
                          fluid_pressure = (-pI * J * F_Inverse_T);


                          drag_lift_value = (stress_fluid + fluid_pressure) *
                                            fe_face_values.normal_vector(q);

                          // 2D-1: 500; 2D-2 and 2D-3: 20 (see Schaefer/Turek
                          // 1996) No multiplication necessary for FSI
                          // benchmarks
                          if (test_case == "2D-1")
                            drag_lift_value *= 500.0;


                          if (adjoint_rhs == "drag")
                            {
                              // extract x-component (drag) and write
                              // this one into the rhs of the dual functional
                              local_rhs(i) +=
                                drag_lift_value[0] * fe_face_values.JxW(q);
                            }
                          else if (adjoint_rhs == "lift")
                            {
                              // extract y-component (lift) and write
                              // this one into the rhs of the dual functional
                              local_rhs(i) +=
                                drag_lift_value[1] * fe_face_values.JxW(q);
                            }


                        } // end i

                    } // end face_n_q_points

                } // end boundary id 80

            } // end face terms


          // Now, we compute the forces that act on the beam. Here,
          // we have two possibilities as already discussed in the paper.
          // We use again the fluid tensor to compute
          // drag and lift:
          if (cell->material_id() == 0)
            {
              for (unsigned int face = 0;
                   face < GeometryInfo<dim>::faces_per_cell;
                   ++face)
                if (cell->neighbor_index(face) != -1)
                  if (cell->material_id() !=
                        cell->neighbor(face)->material_id() &&
                      cell->face(face)->boundary_id() != 80)
                    {
                      fe_face_values.reinit(cell, face);

                      for (unsigned int q = 0; q < n_face_q_points; ++q)
                        {
                          for (unsigned int i = 0; i < dofs_per_cell; ++i)
                            {
                              const Tensor<2, dim> phi_i_grads_v =
                                fe_face_values[velocities].gradient(i, q);
                              const Tensor<2, dim> phi_i_grads_u =
                                fe_face_values[displacements].gradient(i, q);
                              const double phi_i_p =
                                fe_face_values[pressure].value(i, q);

                              Tensor<2, dim> pI;
                              pI[0][0] = phi_i_p;
                              pI[1][1] = phi_i_p;

                              Tensor<2, dim> grad_v = phi_i_grads_v;

                              const Tensor<2, dim> grad_v_T =
                                ALE_Transformations ::get_grad_v_T<dim>(grad_v);

                              const Tensor<2, dim> F = Identity + phi_i_grads_u;

                              const Tensor<2, dim> F_Inverse =
                                ALE_Transformations ::get_F_Inverse<dim>(F);

                              const Tensor<2, dim> F_Inverse_T =
                                ALE_Transformations ::get_F_Inverse_T<dim>(
                                  F_Inverse);

                              const double J =
                                ALE_Transformations ::get_J<dim>(F);

                              const Tensor<2, dim> sigma_ALE = NSE_in_ALE ::
                                get_stress_fluid_except_pressure_ALE<dim>(
                                  density_fluid,
                                  viscosity,
                                  grad_v,
                                  grad_v_T,
                                  F_Inverse,
                                  F_Inverse_T);

                              Tensor<2, dim> stress_fluid;
                              stress_fluid.clear();
                              stress_fluid = (J * sigma_ALE * F_Inverse_T);

                              Tensor<2, dim> fluid_pressure;
                              fluid_pressure.clear();
                              fluid_pressure = (-pI * J * F_Inverse_T);


                              drag_lift_value =
                                (stress_fluid + fluid_pressure) *
                                fe_face_values.normal_vector(q);

                              // 2D-1: 500; 2D-2 and 2D-3: 20 (see
                              // Schaefer/Turek 1996) No multiplication
                              // necessary for FSI benchmarks
                              if (test_case == "2D-1")
                                drag_lift_value *= 500.0;


                              if (adjoint_rhs == "drag")
                                {
                                  // extract x-component (drag) and write
                                  // this one into the rhs of the dual
                                  // functional
                                  local_rhs(i) +=
                                    drag_lift_value[0] * fe_face_values.JxW(q);
                                }
                              else if (adjoint_rhs == "lift")
                                {
                                  // extract y-component (lift) and write
                                  // this one into the rhs of the dual
                                  // functional
                                  local_rhs(i) +=
                                    drag_lift_value[1] * fe_face_values.JxW(q);
                                }


                            } // end i

                        } // end face_n_q_points
                    }

            } // end mat id 0


          cell->get_dof_indices(local_dof_indices);
          constraints_adjoint.distribute_local_to_global(local_rhs,
                                                         local_dof_indices,
                                                         system_rhs_adjoint);

          // Finally, we arrive at the end for assembling
          // the variational formulation for the fluid part and step to
          // the assembling process of the structure terms:
        }
      else if (cell->material_id() == 1)
        {
          // Info: normally, we do not need this for FSI since we assemble
          // the FSI stresses from the fluid side
          // abort();
        }

    } // end cell

  timer.exit_section();
}


template <int dim>
void
FSI_PU_DWR_Problem<dim>::assemble_rhs_adjoint_pressure_point()
{
  timer.enter_section("Assemble adjoint rhs.");
  system_rhs_adjoint = 0;

  Point<dim> evaluation_point(0.15, 0.2);
  Point<dim> evaluation_point_2(0.25, 0.2);

  typename DoFHandler<dim>::active_cell_iterator cell = dof_handler_adjoint
                                                          .begin_active(),
                                                 endc =
                                                   dof_handler_adjoint.end();

  for (; cell != endc; ++cell)
    for (unsigned int vertex = 0; vertex < GeometryInfo<dim>::vertices_per_cell;
         ++vertex)
      {
        if (cell->vertex(vertex).distance(evaluation_point) <
            cell->diameter() * 1e-8)
          {
            // Find the degree of freedom that corresponds to
            // this point in the mesh
            // The first argument is are the vertex coordinates
            // The second argument is the FE component: 0 (vx), 1(vy),
            // 4(pressure)
            system_rhs_adjoint(cell->vertex_dof_index(vertex, 4)) = 1;
          }
        else if (cell->vertex(vertex).distance(evaluation_point_2) <
                 cell->diameter() * 1e-8)
          {
            // Find the degree of freedom that corresponds to
            // this point in the mesh
            // The first argument is are the vertex coordinates
            // The second argument is the FE component: 0 (vx), 1(vy),
            // 4(pressure) The minus is there because the goal functional is a
            // pressure difference
            system_rhs_adjoint(cell->vertex_dof_index(vertex, 4)) = -1;
          }
      }

  timer.exit_section();
}


template <int dim>
void
FSI_PU_DWR_Problem<dim>::assemble_rhs_adjoint_displacement_point()
{
  timer.enter_section("Assemble adjoint rhs.");
  system_rhs_adjoint = 0;

  Point<dim> evaluation_point(0.6, 0.2);

  typename DoFHandler<dim>::active_cell_iterator cell = dof_handler_adjoint
                                                          .begin_active(),
                                                 endc =
                                                   dof_handler_adjoint.end();

  for (; cell != endc; ++cell)
    for (unsigned int vertex = 0; vertex < GeometryInfo<dim>::vertices_per_cell;
         ++vertex)
      {
        if (cell->vertex(vertex).distance(evaluation_point) <
            cell->diameter() * 1e-8)
          {
            // Find the degree of freedom that corresponds to
            // this point in the mesh
            // The first argument is are the vertex coordinates
            // The second argument is the FE component: 0 (vx), 1(vy),
            // 4(pressure) 2: ux, 3: uy
            system_rhs_adjoint(cell->vertex_dof_index(vertex, 2)) = 1;
          }
      }

  timer.exit_section();
}



// Boundary conditions for the linear adjoint problem.
// All non-homogeneous Dirichlet conditions of
// the primal problem are now zer-conditions in the
// adjoint problem.
// All primal Neumann conditions remain adjoint Neumann conditions.
// See optimization books for example.
template <int dim>
void
FSI_PU_DWR_Problem<dim>::set_bc_adjoint()
{
  std::vector<bool> component_mask(dim + dim + 1, true);
  component_mask[dim + dim] = false; // pressure


  VectorTools::interpolate_boundary_values(dof_handler_adjoint,
                                           0,
                                           ZeroFunction<dim>(dim + dim + 1),
                                           constraints_adjoint,
                                           component_mask);
  component_mask[dim] = false; // ux
  VectorTools::interpolate_boundary_values(dof_handler_adjoint,
                                           2,
                                           ZeroFunction<dim>(dim + dim + 1),
                                           constraints_adjoint,
                                           component_mask);

  VectorTools::interpolate_boundary_values(dof_handler_adjoint,
                                           3,
                                           ZeroFunction<dim>(dim + dim + 1),
                                           constraints_adjoint,
                                           component_mask);
  component_mask[dim] = true; // ux
  VectorTools::interpolate_boundary_values(dof_handler_adjoint,
                                           80,
                                           ZeroFunction<dim>(dim + dim + 1),
                                           constraints_adjoint,
                                           component_mask);
  VectorTools::interpolate_boundary_values(dof_handler_adjoint,
                                           81,
                                           ZeroFunction<dim>(dim + dim + 1),
                                           constraints_adjoint,
                                           component_mask);

  // do-nothing outflow condition (Heywood/Rannacher/Turek, 1996)
  component_mask[0] = false;
  component_mask[1] = false;

  VectorTools::interpolate_boundary_values(dof_handler_adjoint,
                                           1,
                                           ZeroFunction<dim>(dim + dim + 1),
                                           constraints_adjoint,
                                           component_mask);
}



// In this function, we solve the linear adjoint system.
// For simplicity we use a direct solver from UMFPACK.
template <int dim>
void
FSI_PU_DWR_Problem<dim>::solve_adjoint()
{
  // Assembling linear adjoint system
  // Matrix is the derivative of the PDE
  //  (i.e., the Jacobian of the primal problem; can be taken
  //  more or less as copy and paste from the assemble_matrix_primal function
  assemble_matrix_adjoint();

  // The rhs depends on the specific goal functional
  if (adjoint_rhs == "drag")
    assemble_rhs_adjoint_drag();
  else if (adjoint_rhs == "pressure")
    assemble_rhs_adjoint_pressure_point();
  else if (adjoint_rhs == "lift")
    assemble_rhs_adjoint_drag(); // drag and lift are assembled in the same
                                 // function
  else if (adjoint_rhs == "displacement")
    assemble_rhs_adjoint_displacement_point();

  // Solving the linear adjoint system
  Timer timer_solve_adjoint;
  timer_solve_adjoint.start();

  // Linear solution
  timer.enter_section("Solve linear adjoint system.");
  Vector<double> sol, rhs;
  sol = solution_adjoint;
  rhs = system_rhs_adjoint;

  SparseDirectUMFPACK A_direct;
  A_direct.factorize(system_matrix_adjoint);

  A_direct.vmult(sol, rhs);
  solution_adjoint = sol;

  constraints_adjoint.distribute(solution_adjoint);
  timer_solve_adjoint.stop();

  std::cout << "Wall time solving adjoint system: " << timer_solve_adjoint()
            << std::endl;

  timer_solve_adjoint.reset();

  timer.exit_section();
}



////////////////////////////////////////////////////////////////////////////////////////////
// 1. Output into vtk
// 2. Evaluation of goal functionals (quantities of interest



// This function is known from almost all other
// tutorial steps in deal.II. This we have different
// finite elements working on the same triangulation, we first
// need to create a joint FE such that we can output all quantities
// together.
template <int dim>
void
FSI_PU_DWR_Problem<dim>::output_results(
  const unsigned int refinement_cycle) const
{
  const FESystem<dim> joint_fe(fe_primal, 1, fe_adjoint, 1);
  DoFHandler<dim>     joint_dof_handler(triangulation);
  joint_dof_handler.distribute_dofs(joint_fe);
  Assert(joint_dof_handler.n_dofs() ==
           dof_handler_primal.n_dofs() + dof_handler_adjoint.n_dofs(),
         ExcInternalError());

  Vector<double> joint_solution(joint_dof_handler.n_dofs());


  {
    std::vector<types::global_dof_index> local_joint_dof_indices(
      joint_fe.dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices_primal(
      fe_primal.dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices_adjoint(
      fe_adjoint.dofs_per_cell);

    typename DoFHandler<dim>::active_cell_iterator
      joint_cell   = joint_dof_handler.begin_active(),
      joint_endc   = joint_dof_handler.end(),
      cell_primal  = dof_handler_primal.begin_active(),
      cell_adjoint = dof_handler_adjoint.begin_active();
    for (; joint_cell != joint_endc;
         ++joint_cell, ++cell_primal, ++cell_adjoint)
      {
        joint_cell->get_dof_indices(local_joint_dof_indices);
        cell_primal->get_dof_indices(local_dof_indices_primal);
        cell_adjoint->get_dof_indices(local_dof_indices_adjoint);

        for (unsigned int i = 0; i < joint_fe.dofs_per_cell; ++i)
          if (joint_fe.system_to_base_index(i).first.first == 0)
            {
              Assert(joint_fe.system_to_base_index(i).second <
                       local_dof_indices_primal.size(),
                     ExcInternalError());
              joint_solution(local_joint_dof_indices[i]) = solution_primal(
                local_dof_indices_primal[joint_fe.system_to_base_index(i)
                                           .second]);
            }
          else
            {
              Assert(joint_fe.system_to_base_index(i).first.first == 1,
                     ExcInternalError());
              Assert(joint_fe.system_to_base_index(i).second <
                       local_dof_indices_adjoint.size(),
                     ExcInternalError());
              joint_solution(local_joint_dof_indices[i]) = solution_adjoint(
                local_dof_indices_adjoint[joint_fe.system_to_base_index(i)
                                            .second]);
            }
      }
  }



  std::vector<std::string> solution_names;
  solution_names.push_back("x_velo");
  solution_names.push_back("y_velo");
  solution_names.push_back("x_dis");
  solution_names.push_back("y_dis");
  solution_names.push_back("p_fluid");

  solution_names.push_back("x_velo_adjoint");
  solution_names.push_back("y_velo_adjoint");
  solution_names.push_back("x_dis_adjoint");
  solution_names.push_back("y_dis_adjoint");
  solution_names.push_back("p_fluid_adjoint");

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation(
      dim + dim + 1 + dim + dim + 1,
      DataComponentInterpretation::component_is_scalar);


  DataOut<dim> data_out;
  data_out.attach_dof_handler(joint_dof_handler);

  data_out.add_data_vector(joint_solution,
                           solution_names,
                           DataOut<dim>::type_dof_data,
                           data_component_interpretation);

  data_out.build_patches();

  std::string filename_basis;
  filename_basis = "solution_fsi_PU_DWR_";

  std::ostringstream filename;

  std::cout << "------------------" << std::endl;
  std::cout << "Write solution" << std::endl;
  std::cout << "------------------" << std::endl;
  std::cout << std::endl;
  filename << filename_basis << Utilities::int_to_string(refinement_cycle, 5)
           << ".vtk";

  std::ofstream output(filename.str().c_str());
  data_out.write_vtk(output);
}

// With help of this function, we extract
// point values for a certain component from our
// discrete solution. We use it to gain the
// displacements of the structure in the x- and y-directions.
template <int dim>
double
FSI_PU_DWR_Problem<dim>::compute_point_value(Point<dim>         p,
                                             const unsigned int component) const
{
  Vector<double> tmp_vector(dim + dim + 1);
  VectorTools::point_value(dof_handler_primal, solution_primal, p, tmp_vector);

  return tmp_vector(component);
}

// Now, we arrive at the function that is responsible
// to compute the line integrals for the drag and the lift. Note, that
// by a proper transformation via the Gauss theorem, the both
// quantities could also be achieved by domain integral computation.
// Nevertheless, we choose the line integration because deal.II provides
// all routines for face value evaluation.
template <int dim>
void
FSI_PU_DWR_Problem<dim>::compute_drag_lift_fsi_fluid_tensor()
{
  const QGauss<dim - 1> face_quadrature_formula(3);
  FEFaceValues<dim>     fe_face_values(fe_primal,
                                   face_quadrature_formula,
                                   update_values | update_gradients |
                                     update_normal_vectors | update_JxW_values);

  const unsigned int dofs_per_cell   = fe_primal.dofs_per_cell;
  const unsigned int n_face_q_points = face_quadrature_formula.size();

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  std::vector<Vector<double>>          face_solution_values(n_face_q_points,
                                                   Vector<double>(dim + dim +
                                                                  1));

  std::vector<std::vector<Tensor<1, dim>>> face_solution_grads(
    n_face_q_points, std::vector<Tensor<1, dim>>(dim + dim + 1));

  Tensor<1, dim> drag_lift_value;

  typename DoFHandler<dim>::active_cell_iterator cell = dof_handler_primal
                                                          .begin_active(),
                                                 endc =
                                                   dof_handler_primal.end();

  for (; cell != endc; ++cell)
    {
      // First, we are going to compute the forces that
      // act on the cylinder. We notice that only the fluid
      // equations are defined here.
      for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
           ++face)
        if (cell->face(face)->at_boundary() &&
            cell->face(face)->boundary_id() == 80)
          {
            fe_face_values.reinit(cell, face);
            fe_face_values.get_function_values(solution_primal,
                                               face_solution_values);
            fe_face_values.get_function_gradients(solution_primal,
                                                  face_solution_grads);

            for (unsigned int q_point = 0; q_point < n_face_q_points; ++q_point)
              {
                const Tensor<2, dim> pI =
                  ALE_Transformations ::get_pI<dim>(q_point,
                                                    face_solution_values);

                const Tensor<2, dim> grad_v =
                  ALE_Transformations ::get_grad_v<dim>(q_point,
                                                        face_solution_grads);

                const Tensor<2, dim> grad_v_T =
                  ALE_Transformations ::get_grad_v_T<dim>(grad_v);

                const Tensor<2, dim> F =
                  ALE_Transformations ::get_F<dim>(q_point,
                                                   face_solution_grads);

                const Tensor<2, dim> F_Inverse =
                  ALE_Transformations ::get_F_Inverse<dim>(F);

                const Tensor<2, dim> F_Inverse_T =
                  ALE_Transformations ::get_F_Inverse_T<dim>(F_Inverse);

                const double J = ALE_Transformations ::get_J<dim>(F);

                const Tensor<2, dim> sigma_ALE =
                  NSE_in_ALE ::get_stress_fluid_except_pressure_ALE<dim>(
                    density_fluid,
                    viscosity,
                    grad_v,
                    grad_v_T,
                    F_Inverse,
                    F_Inverse_T);

                Tensor<2, dim> stress_fluid;
                stress_fluid.clear();
                stress_fluid = (J * sigma_ALE * F_Inverse_T);

                Tensor<2, dim> fluid_pressure;
                fluid_pressure.clear();
                fluid_pressure = (-pI * J * F_Inverse_T);

                drag_lift_value -= (stress_fluid + fluid_pressure) *
                                   fe_face_values.normal_vector(q_point) *
                                   fe_face_values.JxW(q_point);
              }
          } // end boundary 80 for fluid

      // Now, we compute the forces that act on the beam. Here,
      // we have two possibilities as already discussed in the paper.
      // We use again the fluid tensor to compute
      // drag and lift:
      if (cell->material_id() == 0)
        {
          for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
               ++face)
            if (cell->neighbor_index(face) != -1)
              if (cell->material_id() != cell->neighbor(face)->material_id() &&
                  cell->face(face)->boundary_id() != 80)
                {
                  fe_face_values.reinit(cell, face);
                  fe_face_values.get_function_values(solution_primal,
                                                     face_solution_values);
                  fe_face_values.get_function_gradients(solution_primal,
                                                        face_solution_grads);

                  for (unsigned int q_point = 0; q_point < n_face_q_points;
                       ++q_point)
                    {
                      const Tensor<2, dim> pI =
                        ALE_Transformations ::get_pI<dim>(q_point,
                                                          face_solution_values);

                      const Tensor<2, dim> grad_v =
                        ALE_Transformations ::get_grad_v<dim>(
                          q_point, face_solution_grads);

                      const Tensor<2, dim> grad_v_T =
                        ALE_Transformations ::get_grad_v_T<dim>(grad_v);

                      const Tensor<2, dim> F =
                        ALE_Transformations ::get_F<dim>(q_point,
                                                         face_solution_grads);

                      const Tensor<2, dim> F_Inverse =
                        ALE_Transformations ::get_F_Inverse<dim>(F);

                      const Tensor<2, dim> F_Inverse_T =
                        ALE_Transformations ::get_F_Inverse_T<dim>(F_Inverse);

                      const double J = ALE_Transformations ::get_J<dim>(F);

                      const Tensor<2, dim> sigma_ALE =
                        NSE_in_ALE ::get_stress_fluid_except_pressure_ALE<dim>(
                          density_fluid,
                          viscosity,
                          grad_v,
                          grad_v_T,
                          F_Inverse,
                          F_Inverse_T);

                      Tensor<2, dim> stress_fluid;
                      stress_fluid.clear();
                      stress_fluid = (J * sigma_ALE * F_Inverse_T);

                      Tensor<2, dim> fluid_pressure;
                      fluid_pressure.clear();
                      fluid_pressure = (-pI * J * F_Inverse_T);

                      drag_lift_value -= 1.0 * (stress_fluid + fluid_pressure) *
                                         fe_face_values.normal_vector(q_point) *
                                         fe_face_values.JxW(q_point);
                    }
                }
        }
    }

  // 2D-1: 500; 2D-2 and 2D-3: 20 (see Schaefer/Turek 1996)
  // No multiplication necessary for FSI benchmarks
  if (test_case == "2D-1")
    drag_lift_value *= 500.0;

  std::cout << "Face drag:   " << "   " << std::setprecision(16)
            << drag_lift_value[0] << std::endl;
  std::cout << "Face lift:   " << "   " << std::setprecision(16)
            << drag_lift_value[1] << std::endl;

  if (test_case == "2D-1")
    {
      // Stokes: 3.1424267477326167e+00;
      reference_value_drag = 5.5787294556197073e+00; // global ref 4
      reference_value_lift = 1.0610686398307201e-02; // global ref 4
    }
  else if (test_case == "FSI_1")
    {
      reference_value_drag = 1.5370185576528707e+01;
      reference_value_lift = 7.4118844385273164e-01;
    }


  if (adjoint_rhs == "drag")
    {
      exact_error_local = 0.0;
      exact_error_local = std::abs(drag_lift_value[0] - reference_value_drag);
    }
  else if (adjoint_rhs == "lift")
    {
      exact_error_local = 0.0;
      exact_error_local = std::abs(drag_lift_value[1] - reference_value_lift);
    }
}

template <int dim>
void
FSI_PU_DWR_Problem<dim>::compute_drag_lift_fsi_fluid_tensor_domain()
{
  unsigned int drag_lift_select   = 0;
  double       drag_lift_constant = 1.0;

  double value      = 0.0;
  system_rhs_primal = 0;
  const QGauss<dim> quadrature_formula(3);
  FEValues<dim>     fe_values(fe_primal,
                          quadrature_formula,
                          update_values | update_gradients | update_JxW_values |
                            update_q_points);

  const unsigned int dofs_per_cell = fe_primal.dofs_per_cell;
  const unsigned int n_q_points    = quadrature_formula.size();

  const FEValuesExtractors::Vector velocities(0);
  const FEValuesExtractors::Vector displacements(dim);
  const FEValuesExtractors::Scalar pressure(dim + dim);



  Vector<double>                       local_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  std::vector<Vector<double>> old_solution_values(n_q_points,
                                                  Vector<double>(dim + dim +
                                                                 1));

  std::vector<std::vector<Tensor<1, dim>>> old_solution_grads(
    n_q_points, std::vector<Tensor<1, dim>>(dim + dim + 1));

  typename DoFHandler<dim>::active_cell_iterator cell = dof_handler_primal
                                                          .begin_active(),
                                                 endc =
                                                   dof_handler_primal.end();

  for (; cell != endc; ++cell)
    {
      local_rhs = 0;

      fe_values.reinit(cell);
      fe_values.get_function_values(solution_primal, old_solution_values);
      fe_values.get_function_gradients(solution_primal, old_solution_grads);

      for (unsigned int q = 0; q < n_q_points; ++q)
        {
          const Tensor<2, dim> pI =
            ALE_Transformations ::get_pI<dim>(q, old_solution_values);

          const Tensor<1, dim> v =
            ALE_Transformations ::get_v<dim>(q, old_solution_values);

          const Tensor<2, dim> grad_v =
            ALE_Transformations ::get_grad_v<dim>(q, old_solution_grads);

          const Tensor<2, dim> grad_v_T =
            ALE_Transformations ::get_grad_v_T<dim>(grad_v);

          const Tensor<2, dim> F =
            ALE_Transformations ::get_F<dim>(q, old_solution_grads);

          const Tensor<2, dim> F_Inverse =
            ALE_Transformations ::get_F_Inverse<dim>(F);

          const Tensor<2, dim> F_Inverse_T =
            ALE_Transformations ::get_F_Inverse_T<dim>(F_Inverse);

          const double J = ALE_Transformations ::get_J<dim>(F);


          // This is the fluid stress tensor in ALE formulation
          const Tensor<2, dim> sigma_ALE =
            NSE_in_ALE ::get_stress_fluid_except_pressure_ALE<dim>(
              density_fluid,
              viscosity,
              grad_v,
              grad_v_T,
              F_Inverse,
              F_Inverse_T);


          Tensor<2, dim> stress_fluid;
          stress_fluid.clear();
          stress_fluid = (J * sigma_ALE * F_Inverse_T);


          Tensor<2, dim> fluid_pressure;
          fluid_pressure.clear();
          fluid_pressure = (-pI * J * F_Inverse_T);


          Tensor<1, dim> convection_fluid;
          convection_fluid.clear();
          convection_fluid = density_fluid * J * (grad_v * F_Inverse * v);

          // Divergence of the fluid in the ALE formulation
          const double incompressiblity_fluid =
            NSE_in_ALE ::get_Incompressibility_ALE<dim>(q, old_solution_grads);



          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              const unsigned int comp_i =
                fe_primal.system_to_component_index(i).first;

              // Compute drag lift domain tensor
              if (comp_i == 0 || comp_i == 1)
                {
                  const Tensor<1, dim> phi_i_v =
                    fe_values[velocities].value(i, q);
                  const Tensor<2, dim> phi_i_grads_v =
                    fe_values[velocities].gradient(i, q);

                  local_rhs(i) -=
                    (convection_fluid * phi_i_v +
                     scalar_product(fluid_pressure, phi_i_grads_v) +
                     scalar_product(stress_fluid, phi_i_grads_v)) *
                    fe_values.JxW(q);
                }
              else if (comp_i == 2 || comp_i == 3)
                {
                }
              else if (comp_i == 4)
                {
                  const double phi_i_p = fe_values[pressure].value(i, q);
                  local_rhs(i) -=
                    (incompressiblity_fluid * phi_i_p) * fe_values.JxW(q);
                }
            }
        } // end q_points

      cell->get_dof_indices(local_dof_indices);

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        system_rhs_primal(local_dof_indices[i]) += local_rhs(i);

    } // end cell



  std::vector<bool> component_mask(dim + dim + 1, true);
  component_mask[dim]       = false;
  component_mask[dim + 1]   = true;
  component_mask[dim + dim] = true; // pressure

  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(
    dof_handler_primal,
    80,
    ComponentSelectFunction<dim>(drag_lift_select,
                                 drag_lift_constant,
                                 dim + dim + 1),
    boundary_values,
    component_mask);


  VectorTools::interpolate_boundary_values(dof_handler_primal,
                                           0,
                                           ZeroFunction<dim>(dim + dim + 1),
                                           boundary_values,
                                           component_mask);

  VectorTools::interpolate_boundary_values(dof_handler_primal,
                                           1,
                                           ZeroFunction<dim>(dim + dim + 1),
                                           boundary_values,
                                           component_mask);

  VectorTools::interpolate_boundary_values(dof_handler_primal,
                                           2,
                                           ZeroFunction<dim>(dim + dim + 1),
                                           boundary_values,
                                           component_mask);

  VectorTools::interpolate_boundary_values(dof_handler_primal,
                                           81,
                                           ZeroFunction<dim>(dim + dim + 1),
                                           boundary_values,
                                           component_mask);

  value = 0.;

  for (std::map<types::global_dof_index, double>::const_iterator p =
         boundary_values.begin();
       p != boundary_values.end();
       p++)
    {
      value += p->second * system_rhs_primal(p->first);
    }


  global_drag_lift_value += value;
}



template <int dim>
void
FSI_PU_DWR_Problem<dim>::compute_drag_lift_fsi_fluid_tensor_domain_structure()
{
  unsigned int drag_lift_select   = 0;
  double       drag_lift_constant = 1.0;

  double value = 0.0;

  // TODO: check whether system_rhs is good function here and not overwritten
  // in some stupid sense
  system_rhs_primal = 0;
  const QGauss<dim> quadrature_formula(3);
  FEValues<dim>     fe_values(fe_primal,
                          quadrature_formula,
                          update_values | update_gradients | update_JxW_values |
                            update_q_points);

  const unsigned int dofs_per_cell = fe_primal.dofs_per_cell;
  const unsigned int n_q_points    = quadrature_formula.size();

  const FEValuesExtractors::Vector velocities(0);
  const FEValuesExtractors::Vector displacements(dim);
  const FEValuesExtractors::Scalar pressure(dim + dim);



  Vector<double>                       local_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  std::vector<Vector<double>> old_solution_values(n_q_points,
                                                  Vector<double>(dim + dim +
                                                                 1));

  std::vector<std::vector<Tensor<1, dim>>> old_solution_grads(
    n_q_points, std::vector<Tensor<1, dim>>(dim + dim + 1));

  typename DoFHandler<dim>::active_cell_iterator cell = dof_handler_primal
                                                          .begin_active(),
                                                 endc =
                                                   dof_handler_primal.end();

  for (; cell != endc; ++cell)
    {
      local_rhs = 0;

      fe_values.reinit(cell);
      fe_values.get_function_values(solution_primal, old_solution_values);
      fe_values.get_function_gradients(solution_primal, old_solution_grads);

      if (cell->material_id() == 1)
        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            const Tensor<2, dim> F =
              ALE_Transformations ::get_F<dim>(q, old_solution_grads);

            const Tensor<2, dim> F_T = ALE_Transformations ::get_F_T<dim>(F);

            const Tensor<2, dim> Identity =
              ALE_Transformations ::get_Identity<dim>();

            const Tensor<2, dim> F_Inverse =
              ALE_Transformations ::get_F_Inverse<dim>(F);

            const Tensor<2, dim> F_Inverse_T =
              ALE_Transformations ::get_F_Inverse_T<dim>(F_Inverse);

            const double J = ALE_Transformations ::get_J<dim>(F);

            const Tensor<2, dim> E =
              Structure_Terms_in_ALE ::get_E<dim>(F_T, F, Identity);

            const double tr_E = Structure_Terms_in_ALE ::get_tr_E<dim>(E);


            // STVK structure model
            Tensor<2, dim> sigma_structure_ALE;
            sigma_structure_ALE.clear();
            sigma_structure_ALE = (1.0 / J * F *
                                   (lame_coefficient_lambda * tr_E * Identity +
                                    2 * lame_coefficient_mu * E) *
                                   F_T);


            Tensor<2, dim> stress_term;
            stress_term.clear();
            stress_term = (J * sigma_structure_ALE * F_Inverse_T);



            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                const unsigned int comp_i =
                  fe_primal.system_to_component_index(i).first;

                if (comp_i == 0 || comp_i == 1)
                  {
                    const Tensor<2, dim> phi_i_grads_v =
                      fe_values[velocities].gradient(i, q);

                    local_rhs(i) -=
                      (scalar_product(stress_term, phi_i_grads_v)) *
                      fe_values.JxW(q);
                  }
                else if (comp_i == 2 || comp_i == 3)
                  {
                  }
                else if (comp_i == 4)
                  {
                  }
              }
          } // end q_points

      cell->get_dof_indices(local_dof_indices);

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        system_rhs_primal(local_dof_indices[i]) += local_rhs(i);

    } // end cell



  std::vector<bool> component_mask(dim + dim + 1, true);
  component_mask[dim]       = false;
  component_mask[dim + 1]   = true;
  component_mask[dim + dim] = true; // pressure

  std::map<types::global_dof_index, double> boundary_values;

  VectorTools::interpolate_boundary_values(
    dof_handler_primal,
    81,
    ComponentSelectFunction<dim>(drag_lift_select,
                                 drag_lift_constant,
                                 dim + dim + 1),
    boundary_values,
    component_mask);

  VectorTools::interpolate_boundary_values(dof_handler_primal,
                                           80,
                                           ZeroFunction<dim>(dim + dim + 1),
                                           boundary_values,
                                           component_mask);


  VectorTools::interpolate_boundary_values(dof_handler_primal,
                                           0,
                                           ZeroFunction<dim>(dim + dim + 1),
                                           boundary_values,
                                           component_mask);

  VectorTools::interpolate_boundary_values(dof_handler_primal,
                                           1,
                                           ZeroFunction<dim>(dim + dim + 1),
                                           boundary_values,
                                           component_mask);

  VectorTools::interpolate_boundary_values(dof_handler_primal,
                                           2,
                                           ZeroFunction<dim>(dim + dim + 1),
                                           boundary_values,
                                           component_mask);


  value = 0.;

  for (std::map<types::global_dof_index, double>::const_iterator p =
         boundary_values.begin();
       p != boundary_values.end();
       p++)
    {
      value += p->second * system_rhs_primal(p->first);
    }


  global_drag_lift_value += value;
}



template <int dim>
void
FSI_PU_DWR_Problem<dim>::compute_minimal_J()
{
  QGauss<dim>        quadrature_formula(degree + 2);
  FEValues<dim>      fe_values(fe_primal,
                          quadrature_formula,
                          update_values | update_quadrature_points |
                            update_JxW_values | update_gradients);
  const unsigned int n_q_points = quadrature_formula.size();


  std::vector<std::vector<Tensor<1, dim>>> old_solution_grads(
    n_q_points, std::vector<Tensor<1, dim>>(dim + dim + 1));

  double min_J = 1.0e+5;
  double J     = 1.0e+5;


  typename DoFHandler<dim>::active_cell_iterator cell = dof_handler_primal
                                                          .begin_active(),
                                                 endc =
                                                   dof_handler_primal.end();


  for (; cell != endc; ++cell)
    {
      fe_values.reinit(cell);

      fe_values.get_function_gradients(solution_primal, old_solution_grads);

      if (cell->material_id() == 0)
        {
          for (unsigned int q = 0; q < n_q_points; ++q)
            {
              const Tensor<2, dim> F =
                ALE_Transformations ::get_F<dim>(q, old_solution_grads);

              J = ALE_Transformations::get_J<dim>(F);
              if (J < min_J)
                min_J = J;
            }
        }
    }

  std::cout << "Min J: " << "   " << min_J << std::endl;
}



// Here, we compute the four quantities of interest:
// the x and y-displacements of the structure, the drag, and the lift
// and a pressure difference
template <int dim>
void
FSI_PU_DWR_Problem<dim>::compute_functional_values()
{
  double x1, y1, p_front, p_back, p_diff;
  x1 =
    compute_point_value(Point<dim>(0.6, 0.2), dim); // dis-x of flag tip (FSI 1)
  y1 = compute_point_value(Point<dim>(0.6, 0.2),
                           dim + 1); // dis-y of flag tip (FSI 1)

  p_front = compute_point_value(Point<dim>(0.15, 0.2), dim + dim); // pressure
  p_back =
    0.0; // compute_point_value(Point<dim>(0.25,0.2), dim+dim); // pressure

  p_diff = p_front - p_back;

  // Pressure goal functional stuff
  reference_value_p_front = 1.3226157986690765e-01; // global ref 4
  // Stokes: 4.5578330443373852e-02;
  reference_value_p_diff = 1.1750336106835629e-01; // global ref 4
  if (adjoint_rhs == "pressure")
    {
      exact_error_local = 0.0;
      // exact_error_local = std::abs(p_front - reference_value_p_front);
      exact_error_local = std::abs(p_diff - reference_value_p_diff);
    }

  reference_value_flag_tip_ux = 2.2655221197229206e-05;
  reference_value_flag_tip_uy = 8.2018043487426138e-04;

  if (adjoint_rhs == "displacement")
    {
      exact_error_local = 0.0;
      exact_error_local = std::abs(x1 - reference_value_flag_tip_ux);
      // exact_error_local = std::abs(y1 - reference_value_flag_tip_uy);
    }


  std::cout << "------------------" << std::endl;
  std::cout << "DisX  :  " << "   " << std::setprecision(16) << x1 << std::endl;
  std::cout << "DisY  :  " << "   " << std::setprecision(16) << y1 << std::endl;
  std::cout << "P-Diff:  " << "   " << std::setprecision(16) << p_diff
            << std::endl;
  std::cout << "P-front: " << "   " << std::setprecision(16) << p_front
            << std::endl;
  std::cout << "P-back:  " << "   " << std::setprecision(16) << p_back
            << std::endl;
  std::cout << "------------------" << std::endl;

  // Compute drag and lift via line integral
  compute_drag_lift_fsi_fluid_tensor();

  // Compute drag and lift via domain integral
  global_drag_lift_value = 0.0;
  compute_drag_lift_fsi_fluid_tensor_domain();
  compute_drag_lift_fsi_fluid_tensor_domain_structure();
  std::cout << "Domain drag: " << "   " << global_drag_lift_value << std::endl;

  std::cout << "------------------" << std::endl;
  compute_minimal_J();

  std::cout << std::endl;
}


template <int dim>
void
FSI_PU_DWR_Problem<dim>::refine_mesh()
{
  typename DoFHandler<dim>::active_cell_iterator cell = dof_handler_primal
                                                          .begin_active(),
                                                 endc =
                                                   dof_handler_primal.end();

  for (; cell != endc; ++cell)
    {
      // Refine the solid
      if (cell->material_id() == 1)
        cell->set_refine_flag();
    }

  triangulation.execute_coarsening_and_refinement();
}


////////////////////////////////////////////////////////////////////////////////////////////
// PU-DWR Error estimator


// Implementing a weak form of DWR localization using a partition-of-unity
// as it has been proposed in
//
// T. Richter, T. Wick;
// Variational Localizations of the Dual-Weighted Residual Estimator,
// Journal of Computational and Applied Mathematics,
// Vol. 279 (2015), pp. 192-208
//
// Some routines have been taken from step-14 in deal.II
template <int dim>
double
FSI_PU_DWR_Problem<dim>::compute_error_indicators_a_la_PU_DWR(
  const unsigned int refinement_cycle)
{
  // First, we re-initialize the error indicator vector that
  // has the length of the space dimension of the PU-FE.
  // Therein we store the local errors at all degrees of freedom.
  // This is in contrast to usual procedures, where the error
  // in general is stored cell-wise.
  dof_handler_pou.distribute_dofs(fe_pou);
  error_indicators.reinit(dof_handler_pou.n_dofs());


  // Block 1 (building the dual weights):
  // In the following the very specific
  // part (z-I_hz) of DWR is implemented.
  // This part is the same for classical error estimation
  // and PU error estimation.
  std::vector<unsigned int> block_component(5, 0);
  block_component[dim]       = 1;
  block_component[dim + 1]   = 1;
  block_component[dim + dim] = 2;


  DoFRenumbering::component_wise(dof_handler_adjoint, block_component);

  // Implement the interpolation operator
  // (z-z_h)=(z-I_hz)
  ConstraintMatrix dual_hanging_node_constraints;
  DoFTools::make_hanging_node_constraints(dof_handler_adjoint,
                                          dual_hanging_node_constraints);
  dual_hanging_node_constraints.close();

  ConstraintMatrix primal_hanging_node_constraints;
  DoFTools::make_hanging_node_constraints(dof_handler_primal,
                                          primal_hanging_node_constraints);
  primal_hanging_node_constraints.close();


  // TODO double-check (3), 0 , 1, 2 since
  // in the previous code only FSI was implemented
  // Construct a local primal solution that
  // has the length of the adjoint vector
  std::vector<types::global_dof_index> dofs_per_block(3);
  DoFTools::count_dofs_per_block(dof_handler_adjoint,
                                 dofs_per_block,
                                 block_component);
  const unsigned int n_v = dofs_per_block[0];
  const unsigned int n_u = dofs_per_block[1];
  const unsigned int n_p = dofs_per_block[2];


  BlockVector<double> solution_primal_of_adjoint_length;
  solution_primal_of_adjoint_length.reinit(3);
  solution_primal_of_adjoint_length.block(0).reinit(n_v);
  solution_primal_of_adjoint_length.block(1).reinit(n_u);
  solution_primal_of_adjoint_length.block(2).reinit(n_p);
  solution_primal_of_adjoint_length.collect_sizes();

  // Main function 1: Interpolate cell-wise the
  // primal solution into the dual FE space.
  // This rescaled primal solution is called
  //   ** solution_primal_of_adjoint_length **
  FETools::interpolate(dof_handler_primal,
                       solution_primal,
                       dof_handler_adjoint,
                       dual_hanging_node_constraints,
                       solution_primal_of_adjoint_length);



  // Local vectors of dual weights obtained
  // from the adjoint solution
  BlockVector<double> dual_weights;
  dual_weights.reinit(3);
  dual_weights.block(0).reinit(n_v);
  dual_weights.block(1).reinit(n_u);
  dual_weights.block(2).reinit(n_p);
  dual_weights.collect_sizes();

  // Main function 2: Execute (z-I_hz) (in the dual space),
  // yielding the adjoint weights for error estimation.
  FETools::interpolation_difference(dof_handler_adjoint,
                                    dual_hanging_node_constraints,
                                    solution_adjoint,
                                    dof_handler_primal,
                                    primal_hanging_node_constraints,
                                    dual_weights);

  // end Block 1


  // Block 2 (evaluating the PU-DWR):
  // The following function has a loop inside that
  // goes over all cells to collect the error contributions,
  // and is the `heart' of the DWR method. Therein
  // the specific equation of the error estimator is implemented.


  // Info: must be sufficiently high for adjoint evaluations
  QGauss<dim> quadrature_formula(5);

  FEValues<dim> fe_values_pou(fe_pou,
                              quadrature_formula,
                              update_values | update_quadrature_points |
                                update_JxW_values | update_gradients);


  FEValues<dim> fe_values_adjoint(fe_adjoint,
                                  quadrature_formula,
                                  update_values | update_quadrature_points |
                                    update_JxW_values | update_gradients);


  const unsigned int dofs_per_cell = fe_values_pou.dofs_per_cell;
  const unsigned int n_q_points    = fe_values_pou.n_quadrature_points;

  Vector<double> local_err_ind(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);



  const FEValuesExtractors::Vector velocities(0);       // 0
  const FEValuesExtractors::Vector displacements(dim);  // 2
  const FEValuesExtractors::Scalar pressure(dim + dim); // 4

  const FEValuesExtractors::Scalar pou_extract(0);


  std::vector<Vector<double>> primal_cell_values(n_q_points,
                                                 Vector<double>(dim + dim + 1));

  std::vector<std::vector<Tensor<1, dim>>> primal_cell_gradients(
    n_q_points, std::vector<Tensor<1, dim>>(dim + dim + 1));

  std::vector<Vector<double>> dual_weights_values(n_q_points,
                                                  Vector<double>(dim + dim +
                                                                 1));

  std::vector<std::vector<Tensor<1, dim>>> dual_weights_gradients(
    n_q_points, std::vector<Tensor<1, dim>>(dim + dim + 1));

  const Tensor<2, dim> Identity = ALE_Transformations ::get_Identity<dim>();

  typename DoFHandler<dim>::active_cell_iterator cell = dof_handler_pou
                                                          .begin_active(),
                                                 endc = dof_handler_pou.end();

  typename DoFHandler<dim>::active_cell_iterator cell_adjoint =
    dof_handler_adjoint.begin_active();

  for (; cell != endc; ++cell, ++cell_adjoint)
    {
      fe_values_pou.reinit(cell);
      fe_values_adjoint.reinit(cell_adjoint);

      local_err_ind = 0;


      // primal solution (cell residuals)
      // But we use the adjoint FE since we previously enlarged the
      // primal solution to the length of the adjoint vector.
      fe_values_adjoint.get_function_values(solution_primal_of_adjoint_length,
                                            primal_cell_values);

      fe_values_adjoint.get_function_gradients(
        solution_primal_of_adjoint_length, primal_cell_gradients);

      // adjoint weights
      fe_values_adjoint.get_function_values(dual_weights, dual_weights_values);

      fe_values_adjoint.get_function_gradients(dual_weights,
                                               dual_weights_gradients);



      // Gather local error indicators while running
      // of the degrees of freedom of the partition of unity
      // and corresponding quadrature points.
      for (unsigned int q = 0; q < n_q_points; ++q)
        {
          // Right hand side
          Tensor<1, dim> fluid_force;
          fluid_force.clear();
          fluid_force[0] = density_fluid * force_fluid_x;
          fluid_force[1] = density_fluid * force_fluid_y;

          // Primal cell values
          Tensor<2, dim> pI;
          pI[0][0] = primal_cell_values[q](4);
          pI[1][1] = primal_cell_values[q](4);

          Tensor<1, 2> v;
          v.clear();
          v[0] = primal_cell_values[q](0);
          v[1] = primal_cell_values[q](1);

          Tensor<2, dim> grad_v;
          grad_v[0][0] = primal_cell_gradients[q][0][0];
          grad_v[0][1] = primal_cell_gradients[q][0][1];
          grad_v[1][0] = primal_cell_gradients[q][1][0];
          grad_v[1][1] = primal_cell_gradients[q][1][1];

          const Tensor<2, dim> grad_v_T =
            ALE_Transformations ::get_grad_v_T<dim>(grad_v);

          Tensor<2, dim> grad_u;
          grad_u[0][0] = primal_cell_gradients[q][2][0];
          grad_u[0][1] = primal_cell_gradients[q][2][1];
          grad_u[1][0] = primal_cell_gradients[q][3][0];
          grad_u[1][1] = primal_cell_gradients[q][3][1];


          const Tensor<2, dim> F = Identity + grad_u;

          const Tensor<2, dim> F_T = ALE_Transformations ::get_F_T<dim>(F);

          const Tensor<2, dim> F_Inverse =
            ALE_Transformations ::get_F_Inverse<dim>(F);

          const Tensor<2, dim> F_Inverse_T =
            ALE_Transformations ::get_F_Inverse_T<dim>(F_Inverse);

          const double J = ALE_Transformations ::get_J<dim>(F);


          // Adjoint weights
          Tensor<1, dim> dw_v;
          dw_v[0] = dual_weights_values[q](0);
          dw_v[1] = dual_weights_values[q](1);

          double dw_p = dual_weights_values[q](4);

          Tensor<2, dim> grad_dw_v;
          grad_dw_v[0][0] = dual_weights_gradients[q][0][0];
          grad_dw_v[0][1] = dual_weights_gradients[q][0][1];
          grad_dw_v[1][0] = dual_weights_gradients[q][1][0];
          grad_dw_v[1][1] = dual_weights_gradients[q][1][1];

          Tensor<2, dim> grad_dw_u;
          grad_dw_u[0][0] = dual_weights_gradients[q][2][0];
          grad_dw_u[0][1] = dual_weights_gradients[q][2][1];
          grad_dw_u[1][0] = dual_weights_gradients[q][3][0];
          grad_dw_u[1][1] = dual_weights_gradients[q][3][1];



          // Fluid
          const Tensor<2, dim> sigma_ALE =
            NSE_in_ALE ::get_stress_fluid_except_pressure_ALE<dim>(
              density_fluid,
              viscosity,
              grad_v,
              grad_v_T,
              F_Inverse,
              F_Inverse_T);

          Tensor<2, dim> stress_fluid;
          stress_fluid.clear();
          stress_fluid = (J * sigma_ALE * F_Inverse_T);

          Tensor<2, dim> fluid_pressure;
          fluid_pressure.clear();
          fluid_pressure = (-pI * J * F_Inverse_T);

          const double incompressiblity_fluid =
            NSE_in_ALE ::get_Incompressibility_ALE<dim>(q,
                                                        primal_cell_gradients);

          Tensor<1, dim> convection_fluid;
          convection_fluid.clear();
          convection_fluid = density_fluid * J * (grad_v * F_Inverse * v);



          // Solid (STVK structure model)
          const Tensor<2, dim> E =
            Structure_Terms_in_ALE ::get_E<dim>(F_T, F, Identity);

          const double tr_E = Structure_Terms_in_ALE ::get_tr_E<dim>(E);

          Tensor<2, dim> sigma_structure_ALE;
          sigma_structure_ALE.clear();
          sigma_structure_ALE = (1.0 / J * F *
                                 (lame_coefficient_lambda * tr_E * Identity +
                                  2 * lame_coefficient_mu * E) *
                                 F_T);

          Tensor<2, dim> stress_term_solid;
          stress_term_solid.clear();
          stress_term_solid = (J * sigma_structure_ALE * F_Inverse_T);


          // Run over all PU degrees of freedom per cell (namely 4 DoFs for Q1
          // FE-PU)
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              Tensor<2, dim> grad_phi_psi_v;
              grad_phi_psi_v[0][0] =
                fe_values_pou[pou_extract].value(i, q) * grad_dw_v[0][0] +
                dw_v[0] * fe_values_pou[pou_extract].gradient(i, q)[0];
              grad_phi_psi_v[0][1] =
                fe_values_pou[pou_extract].value(i, q) * grad_dw_v[0][1] +
                dw_v[0] * fe_values_pou[pou_extract].gradient(i, q)[1];
              grad_phi_psi_v[1][0] =
                fe_values_pou[pou_extract].value(i, q) * grad_dw_v[1][0] +
                dw_v[1] * fe_values_pou[pou_extract].gradient(i, q)[0];
              grad_phi_psi_v[1][1] =
                fe_values_pou[pou_extract].value(i, q) * grad_dw_v[1][1] +
                dw_v[1] * fe_values_pou[pou_extract].gradient(i, q)[1];

              /*
          // For MMPDE - but alpha_u is very small ...
          // Therefore numerically no influence
              Tensor<2,dim> grad_phi_psi_v;
              grad_phi_psi_v[0][0] = fe_values_pou[pou_extract].value(i,q) *
          grad_dw_v[0][0] + dw_v[0] *
          fe_values_pou[pou_extract].gradient(i,q)[0]; grad_phi_psi_v[0][1] =
          fe_values_pou[pou_extract].value(i,q) * grad_dw_v[0][1] + dw_v[0] *
          fe_values_pou[pou_extract].gradient(i,q)[1]; grad_phi_psi_v[1][0] =
          fe_values_pou[pou_extract].value(i,q) * grad_dw_v[1][0] + dw_v[1] *
          fe_values_pou[pou_extract].gradient(i,q)[0]; grad_phi_psi_v[1][1] =
          fe_values_pou[pou_extract].value(i,q) * grad_dw_v[1][1] + dw_v[1] *
          fe_values_pou[pou_extract].gradient(i,q)[1];
              */

              // double divergence_phi_psi = grad_dw_v[0][0] *
              // cell_data.fe_values_pou[pou_extract].value(i,q) + dw_v[0] *
              // cell_data.fe_values_pou[pou_extract].gradient(i,q)[0]
              //	+ grad_dw_v[1][1] *
              // cell_data.fe_values_pou[pou_extract].value(i,q) + dw_v[1] *
              // cell_data.fe_values_pou[pou_extract].gradient(i,q)[1];



              // Implement the error estimator
              // J(u) - J(u_h) \approx \eta := (f,...) - (\nabla u, ...)
              if (cell->material_id() == 0)
                {
                  // First part: (f,...)
                  local_err_ind(i) += (fluid_force * dw_v *
                                       fe_values_pou[pou_extract].value(i, q)) *
                                      fe_values_pou.JxW(q);

                  // Second part: - (\nabla u, ...)
                  local_err_ind(i) -=
                    (convection_fluid * dw_v *
                       fe_values_pou[pou_extract].value(i, q) +
                     scalar_product(fluid_pressure + stress_fluid,
                                    grad_phi_psi_v) +
                     incompressiblity_fluid * dw_p *
                       fe_values_pou[pou_extract].value(i, q)) *
                    fe_values_pou.JxW(q);
                }
              else if (cell->material_id() == 1)
                {
                  local_err_ind(i) -=
                    (scalar_product(stress_term_solid, grad_phi_psi_v)) *
                    fe_values_pou.JxW(q);
                }
            }

        } // end q_points


      // Write all error contributions
      // in their respective places in the global error vector.
      cell->get_dof_indices(local_dof_indices);
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        error_indicators(local_dof_indices[i]) += local_err_ind(i);


    } // end cell loop for PU FE elements


  // Finally, we eliminate and distribute hanging nodes in the error estimator
  ConstraintMatrix dual_hanging_node_constraints_pou;
  DoFTools::make_hanging_node_constraints(dof_handler_pou,
                                          dual_hanging_node_constraints_pou);
  dual_hanging_node_constraints_pou.close();

  // Distributing the hanging nodes
  dual_hanging_node_constraints_pou.condense(error_indicators);

  // Averaging (making the 'solution' continuous)
  dual_hanging_node_constraints_pou.distribute(error_indicators);

  // end Block 2


  // Block 3 (data and terminal print out)
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler_pou);
  data_out.add_data_vector(error_indicators, "error_ind");
  data_out.build_patches();

  std::ostringstream filename;
  filename << "solution_error_indicators_" << refinement_cycle << ".vtk"
           << std::ends;

  std::ofstream out(filename.str().c_str());
  data_out.write_vtk(out);


  // Print out on terminal
  std::cout << "------------------" << std::endl;
  std::cout << std::setiosflags(std::ios::scientific) << std::setprecision(2);
  std::cout << "   Dofs:                   " << dof_handler_primal.n_dofs()
            << std::endl;
  std::cout << "   Exact error:            " << exact_error_local << std::endl;
  double total_estimated_error = 0.0;
  for (unsigned int k = 0; k < error_indicators.size(); k++)
    total_estimated_error += error_indicators(k);

  // Take the absolute of the estimated error.
  // However, we might check if the signs
  // of the exact error and the estimated error are the same.
  total_estimated_error = std::abs(total_estimated_error);

  std::cout << "   Estimated error (prim): " << total_estimated_error
            << std::endl;

  // From the JCAM paper: compute indicator indices to check
  // effectivity of error estimator.
  double total_estimated_error_absolute_values = 0.0;
  for (unsigned int k = 0; k < error_indicators.size(); k++)
    total_estimated_error_absolute_values += std::abs(error_indicators(k));

  // "ind" things were mainly for paper with Thomas Richter (Richter/Wick; JCAM,
  // 2015)
  //  std::cout << "   Estimated error (ind):  " <<
  //  total_estimated_error_absolute_values << std::endl;

  std::cout << "   Ieff:                   "
            << total_estimated_error / exact_error_local << std::endl;
  // std::cout << "   Iind:                   " <<
  // total_estimated_error_absolute_values/exact_error_local << std::endl;



  // Write everything into a file
  // file.precision(3);
  file << std::setiosflags(std::ios::scientific) << std::setprecision(2);
  file << dof_handler_primal.n_dofs() << "\t";
  file << exact_error_local << "\t";
  file << total_estimated_error << "\t";
  file << total_estimated_error_absolute_values << "\t";
  file << total_estimated_error / exact_error_local << "\t";
  file << total_estimated_error_absolute_values / exact_error_local << "\n";
  file.flush();

  // Write everything into a file gnuplot
  file_gnuplot << std::setiosflags(std::ios::scientific)
               << std::setprecision(2);
  // file_gnuplot.precision(3);
  file_gnuplot << dof_handler_primal.n_dofs() << "\t";
  file_gnuplot << exact_error_local << "\t";
  file_gnuplot << total_estimated_error << "\t";
  file_gnuplot << total_estimated_error_absolute_values << "\n";
  file_gnuplot.flush();


  // end Block 3


  // Block 4
  return total_estimated_error;


  // end Block 4
}


// Refinement strategy and carrying out the actual refinement.
template <int dim>
double
FSI_PU_DWR_Problem<dim>::refine_average_with_PU_DWR(
  const unsigned int refinement_cycle)
{
  // Step 1:
  // Obtain error indicators from PU DWR estimator
  double estimated_DWR_error =
    compute_error_indicators_a_la_PU_DWR(refinement_cycle);

  // Step 2: Choosing refinement strategy
  // Here: averaged refinement
  // Alternatives are in deal.II:
  // refine_and_coarsen_fixed_fraction for example
  for (Vector<float>::iterator i = error_indicators.begin();
       i != error_indicators.end();
       ++i)
    *i = std::fabs(*i);


  const unsigned int                   dofs_per_cell_pou = fe_pou.dofs_per_cell;
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell_pou);

  // Refining all cells that have values above the mean value
  double error_indicator_mean_value = error_indicators.mean_value();

  // Pou cell  later
  typename DoFHandler<dim>::active_cell_iterator cell = dof_handler_pou
                                                          .begin_active(),
                                                 endc = dof_handler_pou.end();

  double error_ind = 0.0;
  // 1.1; for drag and lift and none mesh smoothing
  // 5.0; for pressure difference and maximal mesh smoothing
  double alpha = 0.9;


  for (; cell != endc; ++cell)
    {
      error_ind = 0.0;
      cell->get_dof_indices(local_dof_indices);

      for (unsigned int i = 0; i < dofs_per_cell_pou; ++i)
        {
          error_ind += error_indicators(local_dof_indices[i]);
        }

      // For uniform (global) mesh refinement,
      // just comment the following line
      if (error_ind > alpha * error_indicator_mean_value)
        cell->set_refine_flag();
    }


  triangulation.execute_coarsening_and_refinement();

  return estimated_DWR_error;
}



// As usual, we have to call the run method. It handles
// the output stream to the terminal.
// Finally, we perform the refinement loop of
// the solution process.
template <int dim>
void
FSI_PU_DWR_Problem<dim>::run()
{
  // We set runtime parameters to drive the problem.
  // These parameters could also be read from a parameter file that
  // can be handled by the ParameterHandler object (see step-19)
  set_runtime_parameters();

  // Initialize degrees of freedom
  setup_system_primal();
  setup_system_adjoint();

  std::cout << "\n=============================="
            << "=====================================" << std::endl;
  std::cout << "Parameters\n"
            << "==========\n"
            << "Density fluid:     " << density_fluid << "\n"
            << "Density structure: " << density_structure << "\n"
            << "Viscosity fluid:   " << viscosity << "\n"
            << "alpha_u:           " << alpha_u << "\n"
            << "Lame coeff. mu:    " << lame_coefficient_mu << "\n"
            << "TOL primal Newton: " << lower_bound_newton_residual << "\n"
            << "Max. ref. cycles:  " << max_no_refinement_cycles << "\n"
            << "Max. number DoFs:  " << max_no_degrees_of_freedom << "\n"
            << "TOL DWR estimator: " << TOL_DWR_estimator << "\n"
            << "Goal functional:   " << adjoint_rhs << "\n"
            << std::endl;


  // Refinement loop
  for (unsigned int cycle = 0; cycle < max_no_refinement_cycles; ++cycle)
    {
      std::cout << "\n==============================="
                << "=====================================" << std::endl;
      std::cout << "Refinement cycle " << cycle << ':' << std::endl;



      // Solve problems: primal and adjoint
      newton_iteration_primal();
      if (refinement_strategy == 1)
        solve_adjoint();


      // Compute goal functional values: dx, dy, drag, lift, pressure values
      std::cout << std::endl;
      compute_functional_values();


      // Write solutions into vtk
      output_results(cycle);



      // Mesh refinement
      if (cycle >= 0)
        {
          // Use solution transfer to interpolate solution
          // to the next mesh in order to have a better
          // initial guess for the next refinement level.
          BlockVector<double> tmp_solution_primal;
          tmp_solution_primal = solution_primal;

          SolutionTransfer<dim, BlockVector<double>> solution_transfer(
            dof_handler_primal);
          solution_transfer.prepare_for_coarsening_and_refinement(
            tmp_solution_primal);

          // Choose refinement strategy. The choice
          // of '1' will take the PU DWR estimator.
          double estimated_DWR_error = 0.0;
          if (refinement_strategy == 0)
            triangulation.refine_global(1);
          else if (refinement_strategy == 1)
            estimated_DWR_error = refine_average_with_PU_DWR(cycle);
          else if (refinement_strategy == 2)
            refine_mesh();
          else
            {
              std::cout << "No such refinement strategy. Aborting."
                        << std::endl;
              abort();
            }

          // A practical stopping criterion. Once
          // the a posteriori error estimator (here PU DWR), has
          // been shown to be reliable, we can use the estimated
          // error as stopping criterion; say we want to
          // estimate the drag value up to a error tolerance of 1%

          if (estimated_DWR_error < TOL_DWR_estimator)
            {
              std::cout
                << "Terminating. Goal functional has sufficient accuracy: \n"
                << estimated_DWR_error << std::endl;
              break;
            }


          // Update degrees of freedom after mesh refinement
          if (cycle < max_no_refinement_cycles - 1)
            {
              std::cout << "\n------------------" << std::endl;
              std::cout << "Setup DoFs for next refinement cycle:" << std::endl;

              setup_system_primal();

              if (dof_handler_primal.n_dofs() > max_no_degrees_of_freedom)
                {
                  // Set a sufficiently high number such that enough
                  // computations are done, but the memory of your machine /
                  // cluster is not exceeded.
                  std::cout << "Terminating because max number DoFs exceeded."
                            << std::endl;
                  break;
                }

              setup_system_adjoint();

              solution_transfer.interpolate(tmp_solution_primal,
                                            solution_primal);
            }


        } // end mesh refinement

    } // end refinement cycles
}

// The main function looks almost the same
// as in all other deal.II tuturial steps.
int
main()
{
  try
    {
      deallog.depth_console(0);

      FSI_PU_DWR_Problem<2> fsi_pu_dwr_problem(1);
      fsi_pu_dwr_problem.run();
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
