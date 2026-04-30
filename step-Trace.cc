/* ---------------------------------------------------------------------
*
* Copyright (C) 2021 - 2022 by the deal.II authors
*
* This file is part of the deal.II library.
*
* The deal.II library is free software; you can use it, redistribute
* it, and/or modify it under the terms of the GNU Lesser General
* Public License as published by the Free Software Foundation; either
* version 2.1 of the License, or (at your option) any later version.
* The full text of the license can be found in the file LICENSE.md at
* the top level directory of deal.II.
* 
* ---------------------------------------------------------------------
*/


// @sect3{Include files}


// The first include files have all been treated in previous examples.


#include <assert.h>

#include <deal.II/base/function.h>

#include <deal.II/base/convergence_table.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/geometry_info.h>


#include <deal.II/base/timer.h>




#include <deal.II/dofs/dof_tools.h>


#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values.h>


#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/tria.h>


#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/q_collection.h>


#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_ilu.h>


#include <deal.II/lac/sparse_direct.h>




#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>


#include <fstream>
#include <vector>


#include <deal.II/base/function_signed_distance.h>


#include <deal.II/non_matching/fe_immersed_values.h>
#include <deal.II/non_matching/fe_values.h>
#include <deal.II/non_matching/mesh_classifier.h>
#include <cmath>


namespace StepTrace{
  using namespace dealii;



  const Point<2> &  center = Point<2>(0.5,0);
  const Functions::SignedDistance::Sphere<2> level_set_function_1(center, 1.0);

  const Point<2> point_on_line(0.9, 0.0);
  Tensor<1, 2> normal_vector = []{
    Tensor<1, 2> t;
    t[0] = 1.0;
    t[1] = 0.0;
    return t;
  }();

  const Functions::SignedDistance::Plane<2> level_set_function_2(point_on_line, normal_vector);


  
  template <int dim>
  class LaplaceBeltramiSolver{
    public:
      LaplaceBeltramiSolver();

      void run();

    private:
      void make_grid();

      void setup_discrete_level_set();
      
      void setup_discrete_level_set_2();

      void distribute_dofs();
      
      void initialize_matrices();
      
      void assemble_system();
      
      void solve();
      
      void output_results() const;

      double compute_L2_error() const;
      double compute_interface();
      double compute_interface_2();
      double compute_inside();
      double compute_inside_2();
      double compute_H1_error() const;

      bool face_has_ghost_penalty(const typename Triangulation<dim>::active_cell_iterator &cell,
                                  const unsigned int face_index) const;
      
      const unsigned int fe_degree;

      const Functions::ConstantFunction<dim> rhs_function;
      const Functions::ConstantFunction<dim> boundary_condition;


      Triangulation<dim> triangulation;


      // We need two separate DoFHandlers. The first manages the DoFs for the
      // discrete level set function that describes the geometry of the domain.
      const FE_Q<dim> fe_level_set;
      const FE_Q<dim> fe_level_set_2;
      DoFHandler<dim> level_set_dof_handler;
      DoFHandler<dim> level_set_dof_handler_2;
      Vector<double>  level_set;
      Vector<double>  level_set_2; //the line


      // The second DoFHandler manages the DoFs for the solution of the Poisson
      // equation.
      hp::FECollection<dim> fe_collection;
      DoFHandler<dim>       dof_handler;
      Vector<double>        solution;


      NonMatching::MeshClassifier<dim> mesh_classifier;
      NonMatching::MeshClassifier<dim> mesh_classifier_2;


      SparsityPattern      sparsity_pattern;
      SparseMatrix<double> stiffness_matrix;
      Vector<double>       rhs;
      
      int solve_iter;
      double solve_time;
      double accumulation_time;
      double construction_time;
      double accumulation_time_inside;
      double construction_time_inside;


      const double PI = 3.141592653589793238463;
      int intersected_cells;
  };


  template <int dim>
  class RightHandSide : public Function<dim>{
    public:
      virtual double value(const Point<dim> & p,
                          const unsigned int component = 0) const override;
  };


  template <int dim>
  double RightHandSide<dim>::value(const Point<dim> &p,
                                   const unsigned int /*component*/) const{
    return 3*p(0);
  }




  template <int dim>
  LaplaceBeltramiSolver<dim>::LaplaceBeltramiSolver()
  : fe_degree(1)
  , rhs_function(4.0)
  , boundary_condition(1.0)
  , fe_level_set(fe_degree)
  , fe_level_set_2(fe_degree)
  , level_set_dof_handler(triangulation)
  , level_set_dof_handler_2(triangulation)
  , dof_handler(triangulation)
  , mesh_classifier(level_set_dof_handler, level_set)
  , mesh_classifier_2(level_set_dof_handler_2, level_set_2)
  , intersected_cells(0){
    
  }






  // @sect3{Setting up the Background Mesh}
  // We generate a background mesh with perfectly Cartesian cells. Our domain is
  // a unit disc centered at the origin, so we need to make the background mesh
  // a bit larger than $[-1, 1]^{\text{dim}}$ to completely cover $\Omega$.
  template <int dim>
  void LaplaceBeltramiSolver<dim>::make_grid(){
    std::cout << "Creating background mesh" << std::endl;
    GridGenerator::subdivided_hyper_cube  (triangulation,
                                           3,
                                           -1.0,
                                           2.0);

    //boundary points at (1-0.6)^2-(y^2)=1, so x=1, y=+-root(0.84)
    //GridGenerator::hyper_cube(triangulation, 0.0, 2.0);
    //triangulation.refine_global(1);
  }






  // @sect3{Setting up the Discrete Level Set Function}>
  // The discrete level set function is defined on the whole background mesh.
  // Thus, to set up the DoFHandler for the level set function, we distribute
  // DoFs over all elements in $\mathcal{T}_h$. We then set up the discrete
  // level set function by interpolating onto this finite element space.
  template <int dim>
  void LaplaceBeltramiSolver<dim>::setup_discrete_level_set(){
    std::cout << "Setting up discrete level set function" << std::endl;


    level_set_dof_handler.distribute_dofs(fe_level_set);
    level_set.reinit(level_set_dof_handler.n_dofs());

    VectorTools::interpolate(level_set_dof_handler,
                             level_set_function_1,
                             level_set);
  }

  // attempt for second level set:
  template <int dim>
  void LaplaceBeltramiSolver<dim>::setup_discrete_level_set_2(){
    std::cout << "Setting up second level set function (vertical line)" << std::endl;

    //level_set_dof_handler_2.initialize(triangulation, fe_collection);
    level_set_dof_handler_2.distribute_dofs(fe_level_set_2);
    level_set_2.reinit(level_set_dof_handler_2.n_dofs());

    VectorTools::interpolate(level_set_dof_handler_2,
                             level_set_function_2,
                             level_set_2);
  }



  const double PI                 = 3.141592653589793238463;
  double       angle              = (acos(0.4))*2;
  double       inner_angle        = (2*PI) - angle;
  double       triangle_height    = sqrt(0.84);
  //full area = PI, no need to    make new thing for it
  //naming it pacman area for n   ow because it looks like a pacman
  double       pacman_area        = PI * ((inner_angle)/(2*(PI)));
  double       triangle_area      = (triangle_height)*(0.4);
  double       expected_area      = pacman_area + triangle_area;

  // full perimeter = 2*PI
  double       expected_perimeter = 2*PI * ((inner_angle)/(2*(PI)));

  double       cap_volume         =   2.34892;
  double       cap_area           =   3.96463;

  // @sect3{Setting up the Finite Element Space}
  // To set up the finite element space $V_\Omega^h$, we will use 2 different
  // elements: FE_Q and FE_Nothing. For better readability we define an enum for
  // the indices in the order we store them in the hp::FECollection.
  enum ActiveFEIndex{
    lagrange = 0,
    nothing  = 1
  };



  // Abstracted bisection search for finding the zero point for a given function.
  template <int dim, typename point_type, typename function_type>
  point_type bisection_search(
    point_type point_1,
    point_type point_2,
    const function_type &function,
    const unsigned int iteration_limit,
    const double tolerance){

    // Gets initial point values
    double value_1 = function(point_1);
    double value_2 = function(point_2);

    // Ensures that there actually is a root between these values
    AssertThrow(value_1 * value_2 <= 0.0, ExcMessage("No sign change during bisection search"));

    // Checks if initial points meet criteria and returns them if so
    if(std::abs(value_1) <= tolerance){
      return(point_1);
    }
    if(std::abs(value_2) <= tolerance){
      return(point_2);
    }

    point_type point_mid = 0.5 * (point_1 + point_2);
    for (unsigned int i = 0; i < iteration_limit; ++i)
    {
      // Gets mid point and it's value
      point_mid = 0.5 * (point_1 + point_2);
      double value_mid = function(point_mid);

      // If mid point meets criteria we return it
      if(std::abs(value_mid) <= tolerance){
        return(point_mid);
      }
      // Otherwise we replace one edge point with the mid point and keep iterating
      else if(value_1 * value_mid <=0){
        point_2 = point_mid;
        value_2 = value_mid;
      }
      else{
        point_1 = point_mid;
        value_1 = value_mid;
      }
    }

    // If we ran out of iterations we return what midpoint we have
    return(point_mid);
  }



  // This function takes in a specific cell then goes edge by edge
  // Checking for points where the levelset crosses the edge
  // And builds a list of all such points.
  template <int dim, typename levelSetType>
  std::vector<Point<dim>> find_levelset_intersections(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    const levelSetType &levelset){

    std::vector<Point<dim>> points;

    //Loops over every edge of the cell
    for (unsigned int line_num = 0; line_num < GeometryInfo<dim>::lines_per_cell; ++line_num)
    {
      //Finds points on either end of the edge
      Point<dim> point_1 = cell->line(line_num)->vertex(0);
      Point<dim> point_2 = cell->line(line_num)->vertex(1);

      //Skips over edges that aren't crossed
      double value_1 = levelset.value(point_1);
      double value_2 = levelset.value(point_2);
      if(value_1 * value_2 > 0.0){
        continue;
      }

      //Does a bisection search on the edge to find the zero point for the level set
      Point<dim> intersect_point = bisection_search<dim>(point_1,
                                                          point_2,
                                                          [&](const Point<dim> &p){return levelset.value(p);},
                                                          20,
                                                          1e-6);

      //Adds intersected point to the point list
      points.push_back(intersect_point);
    }

    return points;
  }



  // Given a cell and two levelsets we can detect whether or not
  // The levelsets cross within this cell. We assume that both
  // levelsets already intersect this cell.
  template <int dim, typename levelSetType1, typename levelSetType2>
  bool check_crossing(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    const levelSetType1 &level_set_1,
    const levelSetType2 &level_set_2){
      
    // Calculates the list of points where levelset 1 intersects the cell
    const auto intersect_points = find_levelset_intersections<dim,levelSetType1>(cell,level_set_1);

    // We just chuck out edge cases with multiple crossings or only one edge crossing
    if(intersect_points.size() != 2){
      return false;
    }

    // Calculates the value of these points relative to levelset 2
    double intersection_value_1 = level_set_2.value(intersect_points[0]);
    double intersection_value_2 = level_set_2.value(intersect_points[1]);

    // If there is a sign change between the points then we know the levelsets crossed.
    return(intersection_value_1 * intersection_value_2 <= 0.0);
  }



  // We then use the MeshClassifier to check LocationToLevelSet for each cell in
  // the mesh and tell the DoFHandler to use FE_Q on elements that are inside or
  // intersected, and FE_Nothing on the elements that are outside.
  template <int dim>
  void LaplaceBeltramiSolver<dim>::distribute_dofs(){
    std::cout << "Distributing degrees of freedom" << std::endl;


    fe_collection.push_back(FE_Q<dim>(fe_degree));
    fe_collection.push_back(FE_Nothing<dim>());


    for (const auto &cell : dof_handler.active_cell_iterators()){
      const NonMatching::LocationToLevelSet cell_location_1 =
        mesh_classifier.location_to_level_set(cell);

      const NonMatching::LocationToLevelSet cell_location_2 =
        mesh_classifier_2.location_to_level_set(cell);
      //TraceFEM
      //        if (cell_location == NonMatching::LocationToLevelSet::intersected)
      //         cell->set_active_fe_index(ActiveFEIndex::lagrange);
      //      else
      //         cell->set_active_fe_index(ActiveFEIndex::nothing);
      // if (cell_location_1 == NonMatching::LocationToLevelSet::outside)
      //   cell->set_active_fe_index(ActiveFEIndex::nothing);
      // else
      //   cell->set_active_fe_index(ActiveFEIndex::lagrange);


      //Material Id Map:
      //0: outside either levelset
      //1: inside both levelsets
      //2: inside levelset 1 but intersecting levelset 2
      //3: intersected by levelset 1 and inside levelset 2
      //4: intersected by both levelsets but no crossing
      //5: intersected by both levelsets and crossing
      if(cell_location_1 == NonMatching::LocationToLevelSet::outside || 
         cell_location_2 == NonMatching::LocationToLevelSet::outside){
        cell->set_active_fe_index(ActiveFEIndex::nothing);
        cell->set_material_id(0);
      }
      else if(cell_location_1 == NonMatching::LocationToLevelSet::inside){
        if(cell_location_2 == NonMatching::LocationToLevelSet::inside){
          cell->set_active_fe_index(ActiveFEIndex::lagrange);
          cell->set_material_id(1);
        }
        else{
          cell->set_active_fe_index(ActiveFEIndex::nothing);
          cell->set_material_id(2);
        }
      }
      else{
        if(cell_location_2 == NonMatching::LocationToLevelSet::inside){
          cell->set_active_fe_index(ActiveFEIndex::lagrange);
          cell->set_material_id(3);
        }
        else{
          bool crossing = check_crossing<dim>(cell,level_set_function_1,level_set_function_2);
          if(crossing){
            cell->set_active_fe_index(ActiveFEIndex::nothing);
            cell->set_material_id(5);
          }
          else{
            cell->set_active_fe_index(ActiveFEIndex::nothing);
            cell->set_material_id(4);
          }
        }
      }

    }


    dof_handler.distribute_dofs(fe_collection);
  }



  // A function that can find the best point to approximate the levelset crossing
  // by using a gauss_newton optimization method.
  // Should add a check to make sure the guesses stay within the cell, but that's 
  // a bit annoying and not done here.
  template <int dim, typename levelSetType1, typename levelSetType2>
  Point<dim> gauss_newton_optimize(
    const Point<dim> &initial_point,
    const levelSetType1 &level_set_1,
    const levelSetType2 &level_set_2,
    const unsigned int iteration_limit = 20,
    const double step_tolerance = 1e-6,
    const double distance_tolerance = 1e-6,
    const double matrix_tolerance = 1e-10,
    const double minimum_step_size = 1e-4){

    Point<dim> current_guess = initial_point;

    // Iteratively updates our guess to get closer and closer to zero on both levelsets
    for(unsigned int i=0; i<iteration_limit; ++i){
      // Gets the value of both level sets and the squared distance to both at our current guess
      const double value_1 = level_set_1.value(current_guess);
      const double value_2 = level_set_2.value(current_guess);
      const double distance = value_1*value_1 +
                              value_2*value_2;

      // If our distance is small enough, we accept this point and break
      if(distance < distance_tolerance){
        break;
      }

      // Gets the gradient at our point for both levelsets
      const Tensor<1,dim> gradient_1 = level_set_1.gradient(current_guess);
      const Tensor<1,dim> gradient_2 = level_set_2.gradient(current_guess);

      // Sets up the jacobian
      FullMatrix<double> jacobian(2,dim);
      for(unsigned int col=0; col<dim; ++col){
        jacobian(0,col) = gradient_1[col];
        jacobian(1,col) = gradient_2[col];
      }

      // Evaluates the residuals
      Vector<double> residuals(2);
      residuals[0] = value_1;
      residuals[1] = value_2;

      // Calculates A = J^T J
      FullMatrix<double> A(dim,dim);
      jacobian.Tmmult(A, jacobian);

      // Calculates B = -J^T r(x)
      Vector<double> B(dim);
      jacobian.Tvmult(B, residuals);
      B *= -1.0;

      // If our A matrix is singular we can't solve the system, so we accept our previous point and break
      if(std::abs(A.determinant()) < matrix_tolerance){
        std::cout << "Encountered singular matrix in crossing point location" << std::endl;
        break;
      }

      // Solves the gauss-newton problem (A dx = B)
      Vector<double> step(dim);
      A.gauss_jordan();
      A.vmult(step,B);

      // If our step is small enough then we're not making any progress. We accept our previous point and break
      if(step.l2_norm() < step_tolerance){
        break;
      }

      // Converts the step to a tensor so we can add easier
      Tensor<1,dim> step_tensor;
      for(unsigned int col=0; col<dim; ++col){
        step_tensor[col] = step[col];
      }

      // We do a backwards line search to make sure we actually decrease our distance with this step
      double step_strength = 1;
      bool step_made = false;
      while(step_strength > minimum_step_size){
        // Gets a test point
        Point <dim> test_guess = current_guess + step_strength * step_tensor;
        
        // Evaluates the point's distance to the levelsets
        const double test_value_1  = level_set_1.value(test_guess);
        const double test_value_2  = level_set_2.value(test_guess);
        const double test_distance = test_value_1*test_value_1 + 
                                     test_value_2*test_value_2;

        // If the distance is smaller than before, we accept the step, update our gauss, and break from the line search
        if(test_distance < distance){
          step_made = true;
          current_guess = test_guess;
          break;
        }

        // If the distance isn't smaller, we half our step size and try again
        step_strength *= 0.5;
      }

      // If we shrunk our step size so much that we're not making changes, then we assume that there is no better value
      // in this direction, so we accept the previous point and break.
      if(!step_made){
        break;
      }
    }

    return current_guess;
  }



  // This finds the exact point where two level sets cross within a given cell.
  // Works by making a crude linear approximation of the crossing point before
  // Handing it off to an optimization algorithm to further refine it.
  // Make sure that both levelsets actually cross before using this or else
  // You get garbage out.
  template <int dim, typename levelSetType1, typename levelSetType2>
  Point<dim> find_crossing_point(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    const levelSetType1 &level_set_1,
    const levelSetType2 &level_set_2,
    const double matrix_tolerance = 1e-10){

    // This system only works in 2D currently
    static_assert(dim == 2);
    
    // Get's the points where the levelsets cross the cell edges
    const auto lvs1_intersections = find_levelset_intersections(cell, level_set_1);
    const auto lvs2_intersections = find_levelset_intersections(cell, level_set_2);

    // Ensures the levelsets behave properly at the cell boundaries
    // Should be an unnecessary check
    assert(lvs1_intersections.size() == 2);
    assert(lvs2_intersections.size() == 2);

    // Calculates the differences between the two edge crossing points for each levelset
    const Tensor<1,dim> linear_difference_1 = lvs1_intersections[1] - lvs1_intersections[0];
    const Tensor<1,dim> linear_difference_2 = lvs2_intersections[1] - lvs2_intersections[0];

    // Everything below is a simple linear algebra problem used to find the place where the two
    // levelsets cross under a linear approximation.
    // This is the left hand matrix in the A x = B system
    FullMatrix<double> intersection_matrix(2,2);
    intersection_matrix(0,0) =  linear_difference_1[0];
    intersection_matrix(1,0) =  linear_difference_1[1];
    intersection_matrix(0,1) = -linear_difference_2[0];
    intersection_matrix(1,1) = -linear_difference_2[1];

    // If the matrix is singular there is no solution to this and we can't get an approximate crossing point
    if(std::abs(intersection_matrix.determinant()) < matrix_tolerance){
      throw std::runtime_error("Level set intersection approximation is singular");
    }

    // This is the left hand vector in the A x = B system
    Vector<double> intersection_rhs(2);
    intersection_rhs[0] = lvs2_intersections[0][0] - lvs1_intersections[0][0];
    intersection_rhs[1] = lvs2_intersections[0][1] - lvs1_intersections[0][1];

    // Solves for x, which is a vector or parameters along the segments between the crossing points
    Vector<double> intersection_parameters(2);
    intersection_matrix.gauss_jordan();
    intersection_matrix.vmult(intersection_parameters, intersection_rhs);

    // Get's the actual crude point approximating where the level sets cross
    Point<dim> intersection_point = intersection_parameters[0] * (linear_difference_1) + lvs1_intersections[0];

    // Further refines this guess to get an exact crossing point
    Point<dim> crossing_point = gauss_newton_optimize<dim>(intersection_point,
                                                            level_set_1,
                                                            level_set_2);

    return crossing_point;
  }



  // This will return the edge number that a point lies on, needed for corner cut cell
  // evaluations. This could be done away with by holding onto more information from
  // the initial cell type check.
  template <int dim>
  unsigned int find_edge_from_point(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    const Point<dim> &point,
    const double norm_tolerance = 1e-6){

    // This system only works in 2D currently
    static_assert(dim == 2);
    
    // Loops over every edge to check if the point lies on it or not
    for (unsigned int line_num = 0; line_num < GeometryInfo<dim>::lines_per_cell; ++line_num){
      // Gets the vertices for this edge
      const auto edge = cell->line(line_num);
      const Point<dim> vertex_1 = edge->vertex(0);
      const Point<dim> vertex_2 = edge->vertex(1);

      // Shifts positions to the origin to allow computation
      const Tensor<1,dim> shifted_point = point - vertex_1;
      const Tensor<1,dim> shifted_edge  = vertex_2 - vertex_1;

      // Calculates the projection of the point onto the edge
      const double projection_scale = (shifted_point * shifted_edge) / (shifted_edge.norm_square());
      const Tensor<1,dim> projection = projection_scale * shifted_edge;
      
      // Calculates the normal component of the point relative to the edge
      const Tensor<1,dim> normal = shifted_point - projection;

      // If the normal component is small enough then the point belongs to this edge and we call it here
      if(normal.norm() < norm_tolerance){
        return line_num;
      }
    }

    // If the point doesn't belong to any edge then we have a problem
    throw std::runtime_error("Point not on any edge");
  }

  // This will return the edge number for the edge that lies fully within the levelset,
  // Needed for cross cut cell evaluations.
  template <int dim, typename LevelSetType>
  unsigned int find_inside_edge(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    const LevelSetType &level_set){

    // This system only works in 2D currently
    static_assert(dim == 2);
    
    // Loops over each edge checking if it's vertices are inside
    for (unsigned int line_num = 0; line_num < GeometryInfo<dim>::lines_per_cell; ++line_num){
      // Gets the vertices
      const auto edge = cell->line(line_num);
      const Point<dim> vertex_1 = edge->vertex(0);
      const Point<dim> vertex_2 = edge->vertex(1);

      // Evaluates the vertices
      const double value_1 = level_set.value(vertex_1);
      const double value_2 = level_set.value(vertex_2);

      // If both of the vertices are in side then we found are edge and return its value
      if(value_1 < 0.0 && value_2 < 0.0){
        return line_num;
      }
    }

    throw std::runtime_error("No edge inside the level set");
  }



  // This is a simple function to just count the number of vertices that are inside the
  // levelset for a given cell.
  template <int dim, typename LevelSetType>
  unsigned int number_inside_vertices(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    const LevelSetType &level_set){
    
    unsigned int vertex_count = 0;

    // Loops through each vertex
    for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v){
      const Point<dim> vertex = cell->vertex(v);

      // If the vertex is inside we increase our count
      if(level_set.value(vertex) < 0.0){
        vertex_count += 1;
      }
    }

    return vertex_count;
  }



  // @sect3{Sparsity Pattern}
  // The added ghost penalty results in a sparsity pattern similar to a DG
  // method with a symmetric-interior-penalty term. Thus, we can use the
  // make_flux_sparsity_pattern() function to create it. However, since the
  // ghost-penalty terms only act on the faces in $\mathcal{F}_h$, we can pass
  // in a lambda function that tells make_flux_sparsity_pattern() over which
  // faces the flux-terms appear. This gives us a sparsity pattern with minimal
  // number of entries. When passing a lambda function,
  // make_flux_sparsity_pattern requires us to also pass cell and face coupling
  // tables to it. If the problem was vector-valued, these tables would allow us
  // to couple only some of the vector components. This is discussed in step-46.
  template <int dim>
  void LaplaceBeltramiSolver<dim>::initialize_matrices(){
    std::cout << "Initializing matrices" << std::endl;


    const auto face_has_flux_coupling = [&](const auto &       cell,
                                            const unsigned int face_index){
      return this->face_has_ghost_penalty(cell, face_index);
    };


    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());


    const unsigned int n_components = fe_collection.n_components();
    Table<2, DoFTools::Coupling> cell_coupling(n_components, n_components);
    Table<2, DoFTools::Coupling> face_coupling(n_components, n_components);
    cell_coupling[0][0] = DoFTools::always;
    face_coupling[0][0] = DoFTools::always;


    const AffineConstraints<double> constraints;
    const bool                      keep_constrained_dofs = true;


    DoFTools::make_flux_sparsity_pattern(dof_handler,
                                         dsp,
                                         constraints,
                                         keep_constrained_dofs,
                                         cell_coupling,
                                         face_coupling,
                                         numbers::invalid_subdomain_id,
                                         face_has_flux_coupling);
    sparsity_pattern.copy_from(dsp);


    stiffness_matrix.reinit(sparsity_pattern);
    solution.reinit(dof_handler.n_dofs());
    rhs.reinit(dof_handler.n_dofs());
  }






  // The following function describes which faces are part of the set
  // $\mathcal{F}_h$. That is, it returns true if the face of the incoming cell
  // belongs to the set $\mathcal{F}_h$.
  template <int dim>
  bool LaplaceBeltramiSolver<dim>::face_has_ghost_penalty(
    const typename Triangulation<dim>::active_cell_iterator &cell,
    const unsigned int                                       face_index) const{
    if (cell->at_boundary(face_index))
      return false;


    const NonMatching::LocationToLevelSet cell_location =
      mesh_classifier.location_to_level_set(cell);


    const NonMatching::LocationToLevelSet neighbor_location =
      mesh_classifier.location_to_level_set(cell->neighbor(face_index));




    /*TraceFEM: only internal to intersected
    if (cell_location == NonMatching::LocationToLevelSet::intersected &&
        neighbor_location != NonMatching::LocationToLevelSet::outside)
      return true;


    if (neighbor_location == NonMatching::LocationToLevelSet::intersected &&
        cell_location != NonMatching::LocationToLevelSet::outside)
      return true;
      */
    if (cell_location == NonMatching::LocationToLevelSet::intersected &&
      neighbor_location == NonMatching::LocationToLevelSet::intersected)
      return true; 


    return false;
  }






  // @sect3{Assembling the System}
  template <int dim>
  void LaplaceBeltramiSolver<dim>::assemble_system(){
    std::cout << "Assembling" << std::endl;


    RightHandSide<dim> right_hand_side;


    const unsigned int n_dofs_per_cell = fe_collection[0].dofs_per_cell;
    FullMatrix<double> local_stiffness(n_dofs_per_cell, n_dofs_per_cell);
    Vector<double>     local_rhs(n_dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(n_dofs_per_cell);


    const double ghost_parameter   = 0;
    const double norm_grad_parameter   = 1;
    const double norm_grad_exponent=-1.5;
    //const double nitsche_parameter = 5 * (fe_degree + 1) * fe_degree;


    // Since the ghost penalty is similar to a DG flux term, the simplest way to
    // assemble it is to use an FEInterfaceValues object.
    const QGauss<dim - 1>  face_quadrature(fe_degree + 1);
    FEInterfaceValues<dim> fe_interface_values(fe_collection[0],
                                                face_quadrature,
                                                update_gradients |
                                                  update_JxW_values |
                                                  update_normal_vectors);




    // As we iterate over the cells in the mesh, we would in principle have to
    // do the following on each cell, $T$,
    //
    // 1. Construct one quadrature rule to integrate over the intersection with
    // the domain, $T \cap \Omega$, and one quadrature rule to integrate over
    // the intersection with the boundary, $T \cap \Gamma$.
    // 2. Create FEValues-like objects with the new quadratures.
    // 3. Assemble the local matrix using the created FEValues-objects.
    //
    // To make the assembly easier, we use the class NonMatching::FEValues,
    // which does the above steps 1 and 2 for us. The algorithm @cite saye_2015
    // that is used to generate the quadrature rules on the intersected cells
    // uses a 1-dimensional quadrature rule as base. Thus, we pass a 1D
    // Gauss--Legendre quadrature to the constructor of NonMatching::FEValues.
    // On the non-intersected cells, a tensor product of this 1D-quadrature will
    // be used.
    //
    // As stated in the introduction, each cell has 3 different regions: inside,
    // surface, and outside, where the level set function in each region is
    // negative, zero, and positive. We need an UpdateFlags variable for each
    // such region. These are stored on an object of type
    // NonMatching::RegionUpdateFlags, which we pass to NonMatching::FEValues.
    const QGauss<1> quadrature_1D(fe_degree + 1);
    
    QGauss<dim>     quadrature_formula(fe_degree + 1);
    FEValues<dim> fe_values(fe_collection[0],
                        quadrature_formula,
                        update_values | update_gradients |
                          update_quadrature_points | update_JxW_values);


    NonMatching::RegionUpdateFlags region_update_flags;
    region_update_flags.inside = update_values | update_gradients |
                                  update_JxW_values | update_quadrature_points;
    region_update_flags.surface = update_values | update_gradients |
                                  update_JxW_values | update_quadrature_points |
                                  update_normal_vectors;


    NonMatching::FEValues<dim> non_matching_fe_values(fe_collection,
                                                      quadrature_1D,
                                                      region_update_flags,
                                                      mesh_classifier,
                                                      level_set_dof_handler,
                                                      level_set);


      // As we iterate over the cells, we don't need to do anything on the cells
    // that have FENothing elements. To disregard them we use an iterator
    // filter.
    for (const auto &cell :
          dof_handler.active_cell_iterators() |
            IteratorFilters::ActiveFEIndexEqualTo(ActiveFEIndex::lagrange))
      {
        local_stiffness = 0;
        local_rhs       = 0;


        const double cell_side_interface = cell->minimum_vertex_distance();


        // First, we call the reinit function of our NonMatching::FEValues
        // object. In the background, NonMatching::FEValues uses the
        // MeshClassifier passed to its constructor to check if the incoming
        // cell is intersected. If that is the case, NonMatching::FEValues calls
        // the NonMatching::QuadratureGenerator in the background to create the
        // immersed quadrature rules.
        non_matching_fe_values.reinit(cell);


        // After calling reinit, we can retrieve a dealii::FEValues object with
        // quadrature points that corresponds to integrating over the inside
        // region of the cell. This is the object we use to do the local
        // assembly. This is similar to how hp::FEValues builds dealii::FEValues
        // objects. However, one difference here is that the dealii::FEValues
        // object is returned as an optional. This is a type that wraps an
        // object that may or may not be present. This requires us to add an
        // if-statement to check if the returned optional contains a value,
        // before we use it. This might seem odd at first. Why does the function
        // not just return a reference to a const FEValues<dim>? The reason is
        // that in an immersed method, we have essentially no control of how the
        // cuts occur. Even if the cell is formally intersected: $T \cap \Omega
        // \neq \emptyset$, it might be that the cut is only of floating point
        // size $|T \cap \Omega| \sim \epsilon$. When this is the case, we can
        // not expect that the algorithm that generates the quadrature rule
        // produces anything useful. It can happen that the algorithm produces 0
        // quadrature points. When this happens, the returned optional will not
        // contain a value, even if the cell is formally intersected.
        
        /* TraceFEM:
        const std_cxx17::optional<FEValues<dim>> &inside_fe_values =
          non_matching_fe_values.get_inside_fe_values();


        if (inside_fe_values)
          for (const unsigned int q :
                inside_fe_values->quadrature_point_indices())
            {
              const Point<dim> &point = inside_fe_values->quadrature_point(q);
              for (const unsigned int i : inside_fe_values->dof_indices())
                {
                  for (const unsigned int j : inside_fe_values->dof_indices())
                    {
                      local_stiffness(i, j) +=
                        inside_fe_values->shape_grad(i, q) *
                        inside_fe_values->shape_grad(j, q) *
                        inside_fe_values->JxW(q);
                    }
                  local_rhs(i) += rhs_function.value(point) *
                                  inside_fe_values->shape_value(i, q) *
                                  inside_fe_values->JxW(q);
                }
            }
    */
      //TraceFEM: normal-gradient-over-inside stabilization...
          /*const std_cxx17::optional<FEValues<dim>> &intersected_fe_values =
            non_matching_fe_values.get_intersected_fe_values(); //Doesn't exist!
    */
    const std::optional<NonMatching::FEImmersedSurfaceValues<dim>>
          &surface_fe_values = non_matching_fe_values.get_surface_fe_values();


        if (surface_fe_values) {
          typename DoFHandler<dim>::active_cell_iterator level_set_cell(&(triangulation), cell->level(), cell->index(), &level_set_dof_handler);
            fe_values.reinit(level_set_cell);
      std::vector<Tensor<1, dim>> normal(quadrature_formula.size());
            fe_values.get_function_gradients(level_set, normal);   
        
          for (const unsigned int q :
                fe_values.quadrature_point_indices())
            {
              //const Point<dim> &point = fe_values.quadrature_point(q);
              normal[q]=(1.0/normal[q].norm())*normal[q];
              for (const unsigned int i : fe_values.dof_indices())
                {
                  for (const unsigned int j : fe_values.dof_indices())
                    {
                      local_stiffness(i, j) +=
                        norm_grad_parameter * std::pow(cell_side_interface,norm_grad_exponent) * (normal[q] * fe_values.shape_grad(i, q)) * (normal[q] * fe_values.shape_grad(j, q))
                        * fe_values.JxW(q);
                    }
                }
            }
          }
        //...TraceFEM*/
          // In the same way, we can use NonMatching::FEValues to retrieve an
        // FEFaceValues-like object to integrate over $T \cap \Gamma$. The only
        // thing that is new here is the type of the object. The transformation
        // from quadrature weights to JxW-values is different for surfaces, so
        // we need a new class: NonMatching::FEImmersedSurfaceValues. In
        // addition to the ordinary functions shape_value(..), shape_grad(..),
        // etc., one can use its normal_vector(..)-function to get an outward
        // normal to the immersed surface, $\Gamma$. In terms of the level set
        // function, this normal reads
        // @f{equation*}
        //   n = \frac{\nabla \psi}{\| \nabla \psi \|}.
        // @f}
        // An additional benefit of std::optional is that we do not need any
        // other check for whether we are on intersected cells: In case we are
        // on an inside cell, we get an empty object here.
      
        /*const std_cxx17::optional<NonMatching::FEImmersedSurfaceValues<dim>>
          &surface_fe_values = non_matching_fe_values.get_surface_fe_values();
      */
        if (surface_fe_values)
          {
            for (const unsigned int q :
                  surface_fe_values->quadrature_point_indices())
              {
                const Point<dim> &point =
                  surface_fe_values->quadrature_point(q);
                const Tensor<1, dim> &normal =
                  surface_fe_values->normal_vector(q);
                for (const unsigned int i : surface_fe_values->dof_indices())
                  {
                    for (const unsigned int j :
                          surface_fe_values->dof_indices())
                      {
                        local_stiffness(i, j) += (
                          /*TraceFEM -normal * surface_fe_values->shape_grad(i, q) *
                              surface_fe_values->shape_value(j, q) +
                            -normal * surface_fe_values->shape_grad(j, q) *
                              surface_fe_values->shape_value(i, q) +
                            nitsche_parameter / cell_side_interface * */
                              surface_fe_values->shape_value(i, q) *
                              surface_fe_values->shape_value(j, q) +
                                (surface_fe_values->shape_grad(i, q)-(normal*surface_fe_values->shape_grad(i, q))*normal) *
                              (surface_fe_values->shape_grad(j, q)-(normal*surface_fe_values->shape_grad(j, q))*normal)
                              ) *
                          surface_fe_values->JxW(q);
                          
                      }
                      
                    local_rhs(i) += right_hand_side.value(point) *surface_fe_values->shape_value(i, q) * surface_fe_values->JxW(q);
                    /*TraceFEM
                    local_rhs(i) +=
                        boundary_condition.value(point) *
                      (nitsche_parameter / cell_side_interface *
                          surface_fe_values->shape_value(i, q) -
                        normal * surface_fe_values->shape_grad(i, q)) *
                      surface_fe_values->JxW(q);*/
                  }
              }
          }


      
        cell->get_dof_indices(local_dof_indices);


        stiffness_matrix.add(local_dof_indices, local_stiffness);
        rhs.add(local_dof_indices, local_rhs);


      
        // The assembly of the ghost penalty term is straight forward. As we
        // iterate over the local faces, we first check if the current face
        // belongs to the set $\mathcal{F}_h$. The actual assembly is simple
        // using FEInterfaceValues. Assembling in this we will traverse each
        // internal face in the mesh twice, so in order to get the penalty
        // constant we expect, we multiply the penalty term with a factor 1/2.
        
        for (unsigned int f : cell->face_indices())
          if (face_has_ghost_penalty(cell, f))
            {
              const unsigned int invalid_subface =
                numbers::invalid_unsigned_int;




              fe_interface_values.reinit(cell,
                                          f,
                                          invalid_subface,
                                          cell->neighbor(f),
                                          cell->neighbor_of_neighbor(f),
                                          invalid_subface);


              const unsigned int n_interface_dofs =
                fe_interface_values.n_current_interface_dofs();
              FullMatrix<double> local_stabilization(n_interface_dofs,
                                                      n_interface_dofs);
              for (unsigned int q = 0;
                    q < fe_interface_values.n_quadrature_points;
                    ++q)
                {
                  const Tensor<1, dim> normal = fe_interface_values.normal(q);
                  for (unsigned int i = 0; i < n_interface_dofs; ++i)
                    for (unsigned int j = 0; j < n_interface_dofs; ++j)
                      {
                        local_stabilization(i, j) +=
                          .5 * ghost_parameter
                          //TraceFEM: * cell_side_interface
                          * normal *
                          fe_interface_values.jump_in_shape_gradients(i, q) *
                          normal *
                          fe_interface_values.jump_in_shape_gradients(j, q) *
                          fe_interface_values.JxW(q);
                      }
                }


              const std::vector<types::global_dof_index>
                local_interface_dof_indices =
                  fe_interface_values.get_interface_dof_indices();


              stiffness_matrix.add(local_interface_dof_indices,
                                    local_stabilization);
            }
            
            
      }
  }




  // @sect3{Solving the System}
  template <int dim>
  void LaplaceBeltramiSolver<dim>::solve(){
    std::string flag="Direct";
    if (flag=="Direct")
    {
      std::cout << "Solving directly... ";
      Timer timer;
      SparseDirectUMFPACK A_direct;
      A_direct.initialize(stiffness_matrix);
  
      A_direct.vmult(solution, rhs);
      timer.stop();
      std::cout << "took (" << timer.cpu_time() << "s)" << std::endl;
      solve_iter=-1;
      solve_time=timer.cpu_time();
    }
    else if (flag=="PCG-ILU")
      {
      std::cout << "Solving PCG-ILU" << std::endl;
      
      Timer timer;
      const unsigned int max_iterations = solution.size();
      SolverControl      solver_control(max_iterations);
      SolverCG<>         solver(solver_control);
      SparseILU<double> ILU;
      ILU.initialize(stiffness_matrix);
      
      solver.solve(stiffness_matrix, solution, rhs, ILU);
      
      timer.stop();
      
      //cout<<" Number of iter:\t" << solver_control.last_step() << "\n";
      solve_iter=solver_control.last_step();
      solve_time=timer.cpu_time();
      std::cout << "took (" << timer.cpu_time() << "s)" << std::endl;
    }
    else if (flag=="PCG-Jacobi")
      {
      std::cout << "Solving PCG-Jacobi" << std::endl;
      
      Timer timer;
      const unsigned int max_iterations = solution.size();
      SolverControl      solver_control(max_iterations);
      SolverCG<>         solver(solver_control);
      
      PreconditionJacobi<SparseMatrix<double> > Jacobi;
      Jacobi.initialize(stiffness_matrix, PreconditionJacobi<SparseMatrix<double>>::AdditionalData(.6));
      
      solver.solve(stiffness_matrix, solution, rhs, Jacobi);
      
      timer.stop();
      
      //cout<<" Number of iter:\t" << solver_control.last_step() << "\n";
      solve_iter=solver_control.last_step();
      solve_time=timer.cpu_time();
      std::cout << "took (" << timer.cpu_time() << "s)" << std::endl;
    }
    else if (flag=="CG")
      {
      std::cout << "Solving CG" << std::endl;
      Timer timer;
      const unsigned int max_iterations = solution.size();
      SolverControl      solver_control(max_iterations);
      SolverCG<>         solver(solver_control);
      
      solver.solve(stiffness_matrix, solution, rhs, PreconditionIdentity());


      timer.stop();
      
      solve_iter=solver_control.last_step();
      solve_time=timer.cpu_time();
      std::cout << "took (" << timer.cpu_time() << "s)" << std::endl;
    }
    else return;
  }




  // @sect3{Data Output}
  // Since both DoFHandler instances use the same triangulation, we can add both
  // the level set function and the solution to the same vtu-file. Further, we
  // do not want to output the cells that have LocationToLevelSet value outside.
  // To disregard them, we write a small lambda function and use the
  // set_cell_selection function of the DataOut class.
  template <int dim>
  void LaplaceBeltramiSolver<dim>::output_results() const{
    std::cout << "Writing vtu file" << std::endl;


    DataOut<dim> data_out;
    //data_out.add_data_vector(dof_handler, solution, "solution");
    data_out.add_data_vector(level_set_dof_handler, level_set, "level_set");
    data_out.add_data_vector(level_set_dof_handler_2, level_set_2, "level_set_2");

    // Add material ID as cell data
    Vector<float> material_ids(triangulation.n_active_cells());

    unsigned int index = 0;
    for (const auto &cell : triangulation.active_cell_iterators())
    {
      material_ids[index++] = cell->material_id();
    }

    data_out.add_data_vector(material_ids, "material_id");

    data_out.set_cell_selection(
      [this](const typename Triangulation<dim>::cell_iterator &cell) {
        return cell->is_active() &&
              mesh_classifier.location_to_level_set(cell) !=
                NonMatching::LocationToLevelSet::outside &&
              mesh_classifier_2.location_to_level_set(cell) !=
                NonMatching::LocationToLevelSet::outside;
      });
    


    data_out.build_patches();
    std::ofstream output("step-Trace-NEWEST.vtu");
    data_out.write_vtu(output);
  }






  // @sect3{$L^2$-Error}
  // To test that the implementation works as expected, we want to compute the
  // error in the solution in the $L^2$-norm. The analytical solution to the
  // Poisson problem stated in the introduction reads
  // @f{align*}
  //  u(x) = 1 - \frac{2}{\text{dim}}(\| x \|^2 - 1) , \qquad x \in
  //  \overline{\Omega}.
  // @f}
  // We first create a function corresponding to the analytical solution:
  template <int dim>
  class AnalyticalSolution : public Function<dim>{
    public:
      double value(const Point<dim> & point,
                    const unsigned int component = 0) const override;
      Tensor<1,dim>  gradG(const Point<dim> & point,
                    const unsigned int component = 0) const;
  };






  template <int dim>
  double AnalyticalSolution<dim>::value(const Point<dim> & point,
                                        const unsigned int component) const{
    AssertIndexRange(component, this->n_components);
    (void)component;


    return point[0];
    //return 1. - 2. / dim * (point.norm_square() - 1.);
  }



  template <int dim>
  Tensor<1,dim> AnalyticalSolution<dim>::gradG(const Point<dim> & point,
                                        const unsigned int component) const{
    AssertIndexRange(component, this->n_components);
    (void)component;


    Tensor<1,dim> temp;
    temp[0]=1-point[0]*point[0];
    temp[1]=-point[0]*point[1];
    //temp[2]=-point[0]*point[2];
      return temp;
  }






  // Of course, the analytical solution, and thus also the error, is only
  // defined in $\overline{\Omega}$. Thus, to compute the $L^2$-error we must
  // proceed in the same way as when we assembled the linear system. We first
  // create an NonMatching::FEValues object.
  template <int dim>
  double LaplaceBeltramiSolver<dim>::compute_L2_error() const{
    std::cout << "Computing L2 error" << std::endl;


    const QGauss<1> quadrature_1D(fe_degree + 1);


    NonMatching::RegionUpdateFlags region_update_flags;
    region_update_flags.inside =
      update_values | update_JxW_values | update_quadrature_points;
        region_update_flags.surface = update_values | update_gradients |
                                  update_JxW_values | update_quadrature_points |
                                  update_normal_vectors;
                                  
    NonMatching::FEValues<dim> non_matching_fe_values(fe_collection,
                                                      quadrature_1D,
                                                      region_update_flags,
                                                      mesh_classifier,
                                                      level_set_dof_handler,
                                                      level_set);


    // We then iterate iterate over the cells that have LocationToLevelSetValue
    // value inside or intersected again. For each quadrature point, we compute
    // the pointwise error and use this to compute the integral.
    const AnalyticalSolution<dim> analytical_solution;
    double                        error_L2_squared = 0;


    for (const auto &cell :
          dof_handler.active_cell_iterators() |
            IteratorFilters::ActiveFEIndexEqualTo(ActiveFEIndex::lagrange))
      {
      
        non_matching_fe_values.reinit(cell);
        
        /*TraceFEM
        const std_cxx17::optional<FEValues<dim>> &fe_values =
          non_matching_fe_values.get_inside_fe_values();


        if (fe_values)
          {
            std::vector<double> solution_values(fe_values->n_quadrature_points);
            fe_values->get_function_values(solution, solution_values);


            for (const unsigned int q : fe_values->quadrature_point_indices())
              {
                const Point<dim> &point = fe_values->quadrature_point(q);
                const double      error_at_point =
                  solution_values.at(q) - analytical_solution.value(point);
                error_L2_squared +=
                  std::pow(error_at_point, 2) * fe_values->JxW(q);
              }
          }
          */
          
            const std::optional<NonMatching::FEImmersedSurfaceValues<dim>>
          &surface_fe_values = non_matching_fe_values.get_surface_fe_values();


        if (surface_fe_values)
          {
          
          std::vector<double> solution_values(surface_fe_values->n_quadrature_points);
            surface_fe_values->get_function_values(solution, solution_values);


            for (const unsigned int q : surface_fe_values->quadrature_point_indices())
              {
                const Point<dim> &point = surface_fe_values->quadrature_point(q);
                const double      error_at_point =
                  solution_values.at(q) - analytical_solution.value(point);
                error_L2_squared +=
                  std::pow(error_at_point, 2) * surface_fe_values->JxW(q);
              }
              
          }
      }


    return std::sqrt(error_L2_squared);
  }


  // Of course, the analytical solution, and thus also the error, is only
  // defined in $\overline{\Omega}$. Thus, to compute the $L^2$-error we must
  // proceed in the same way as when we assembled the linear system. We first
  // create an NonMatching::FEValues object.
  template <int dim>
  double LaplaceBeltramiSolver<dim>::compute_H1_error() const{
    std::cout << "Computing H1 error" << std::endl;


    const QGauss<1> quadrature_1D(fe_degree + 1);


    NonMatching::RegionUpdateFlags region_update_flags;
    region_update_flags.inside =
      update_values | update_JxW_values | update_quadrature_points;
        region_update_flags.surface = update_values | update_gradients |
                                  update_JxW_values | update_quadrature_points |
                                  update_normal_vectors;
                                  
    NonMatching::FEValues<dim> non_matching_fe_values(fe_collection,
                                                      quadrature_1D,
                                                      region_update_flags,
                                                      mesh_classifier,
                                                      level_set_dof_handler,
                                                      level_set);


    // We then iterate iterate over the cells that have LocationToLevelSetValue
    // value inside or intersected again. For each quadrature point, we compute
    // the pointwise error and use this to compute the integral.
    const AnalyticalSolution<dim> analytical_solution;
    double                        error_H1_squared = 0;


    for (const auto &cell :
          dof_handler.active_cell_iterators() |
            IteratorFilters::ActiveFEIndexEqualTo(ActiveFEIndex::lagrange))
      {
      
        non_matching_fe_values.reinit(cell);
        
                
            const std::optional<NonMatching::FEImmersedSurfaceValues<dim>>
          &surface_fe_values = non_matching_fe_values.get_surface_fe_values();


        if (surface_fe_values)
          {
          
          std::vector<double> solution_values(surface_fe_values->n_quadrature_points);
            surface_fe_values->get_function_values(solution, solution_values);
            


            std::vector<Tensor<1, dim>> solution_grads(surface_fe_values->n_quadrature_points);
            surface_fe_values->get_function_gradients(solution, solution_grads);


            for (const unsigned int q : surface_fe_values->quadrature_point_indices())
              {
                const Point<dim> &point = surface_fe_values->quadrature_point(q);
                const Tensor<1, dim> &normal = surface_fe_values->normal_vector(q);
                const double      error_at_point =
                  (solution_grads.at(q)-(normal*solution_grads.at(q))*normal -
                  analytical_solution.gradG(point))
                  *(solution_grads.at(q)-(normal*solution_grads.at(q))*normal -
                  analytical_solution.gradG(point));
                error_H1_squared +=
                  error_at_point * surface_fe_values->JxW(q);
              }
              
          }
      }


    return std::sqrt(error_H1_squared);
  }



  template <int dim>
  double LaplaceBeltramiSolver<dim>::compute_interface(){
    const QGauss<1> quadrature_1D(fe_degree + 1);

    NonMatching::RegionUpdateFlags region_update_flags;
    region_update_flags.surface = update_JxW_values | update_quadrature_points;

    NonMatching::FEValues<dim> non_matching_fe_values(
      fe_collection,
      quadrature_1D,
      region_update_flags,
      mesh_classifier,
      level_set_dof_handler,
      level_set);

    double interface = 0.0;

    for (const auto &cell : dof_handler.active_cell_iterators() |
                            IteratorFilters::ActiveFEIndexEqualTo(ActiveFEIndex::lagrange))
    {
    //not skipping cells anymore
      non_matching_fe_values.reinit(cell);

      const std::optional<NonMatching::FEImmersedSurfaceValues<dim>> &surface_fe_values =
        non_matching_fe_values.get_surface_fe_values();

      if (surface_fe_values)
      {
        for (const unsigned int q : surface_fe_values->quadrature_point_indices())
        {
          const Point<dim> &point = surface_fe_values->quadrature_point(q);
          const double ls2_value = level_set_function_2.value(point); //level set 2 value of the quadrature point

          if (ls2_value <= 0.0) // check if it's inside level set 2
            interface += surface_fe_values->JxW(q);
        }
      }
    }

    return interface;
  }

//} //this brace may be wrong

 /*template <int dim>
 double LaplaceBeltramiSolver<dim>::compute_interface_2()
 {
     const QGauss<1> quadrature_1D(fe_degree + 1);  
     
     NonMatching::RegionUpdateFlags region_update_flags;
     region_update_flags.surface = update_JxW_values | update_quadrature_points;
 
     
     NonMatching::FEValues<dim> non_matching_fe_values(
         fe_collection,
         quadrature_1D,
         region_update_flags,
         mesh_classifier_2,
         level_set_dof_handler_2,
         level_set_2
     );
 
     double interface_2 = 0.0;  
 
    
     for (const auto &cell : dof_handler.active_cell_iterators() |
                              IteratorFilters::ActiveFEIndexEqualTo(ActiveFEIndex::lagrange))
     {
         // check if cell is outside the level set, and skip it if it is
         if (mesh_classifier.location_to_level_set(cell) == NonMatching::LocationToLevelSet::outside) {
             continue;  // Skip cells outside the level set
         }
 
         non_matching_fe_values.reinit(cell); 
       
         const std::optional<NonMatching::FEImmersedSurfaceValues<dim>> &surface_fe_values = non_matching_fe_values.get_surface_fe_values();
 
         
         if (surface_fe_values)
         {
            
             for (const unsigned int q : surface_fe_values->quadrature_point_indices())
                 interface_2 += surface_fe_values->JxW(q);  
         }
     }
 
     return interface_2; 
 }
 
 
*/

  template <int dim>
  double LaplaceBeltramiSolver<dim>::compute_inside(){
    const QGauss<1> quadrature_1D(fe_degree + 1);

    NonMatching::RegionUpdateFlags region_update_flags;
    region_update_flags.inside = update_JxW_values | update_quadrature_points;

    NonMatching::FEValues<dim> non_matching_fe_values(
      fe_collection,
      quadrature_1D,
      region_update_flags,
      mesh_classifier,
      level_set_dof_handler,
      level_set);

    double inside = 0.0;

    for (const auto &cell : dof_handler.active_cell_iterators() |
                            IteratorFilters::ActiveFEIndexEqualTo(ActiveFEIndex::lagrange))
    {
      non_matching_fe_values.reinit(cell);

      const std::optional<FEValues<dim>> &fe_values =
        non_matching_fe_values.get_inside_fe_values();

      if (fe_values)
      {
        for (const unsigned int q : fe_values->quadrature_point_indices())
        {
          const Point<dim> &point = fe_values->quadrature_point(q);
          const double ls2_value = level_set_function_2.value(point); //value of level set at quadrature point

          if (ls2_value <= 0.0) // check if it's inside level set
            inside += fe_values->JxW(q);
        }
      }
    }

    return inside;
  }


  template <int dim>
  double LaplaceBeltramiSolver<dim>::compute_inside_2(){
    double inside_2 = 0.0;

    
    for (const auto &cell : dof_handler.active_cell_iterators() |
                              IteratorFilters::ActiveFEIndexEqualTo(ActiveFEIndex::lagrange))
    {
        // check if the cell is outside the level set
        if (mesh_classifier.location_to_level_set(cell) == NonMatching::LocationToLevelSet::outside) {
            continue;  
        }

        
        const QGauss<1> quadrature_1D(fe_degree + 1);
        NonMatching::RegionUpdateFlags region_update_flags;
        region_update_flags.inside = update_JxW_values | update_quadrature_points;
        
        NonMatching::FEValues<dim> non_matching_fe_values(
            fe_collection,
            quadrature_1D,
            region_update_flags,
            mesh_classifier_2,
            level_set_dof_handler_2,
            level_set_2
        );
        
        non_matching_fe_values.reinit(cell);  
        

        const std::optional<FEValues<dim>> &fe_values = non_matching_fe_values.get_inside_fe_values();

        if (fe_values)
        {
            
            for (const unsigned int q : fe_values->quadrature_point_indices())
            {
                inside_2 += fe_values->JxW(q);
            }
        }
    }

    return inside_2;
  }
 



  // Of course, the analytical solution, and thus also the error, is only
  // defined in $\overline{\Omega}$. Thus, to compute the $L^2$-error we must
  // proceed in the same way as when we assembled the linear system. We first
  // create an NonMatching::FEValues object.




  // @sect3{A Convergence Study}
  // Finally, we do a convergence study to check that the $L^2$-error decreases
  // with the expected rate. We refine the background mesh a few times. In each
  // refinement cycle, we solve the problem, compute the error, and add the
  // $L^2$-error and the mesh size to a ConvergenceTable.
  template <int dim>
  void LaplaceBeltramiSolver<dim>::run(){
    ConvergenceTable   convergence_table;
    const unsigned int n_refinements = 7;


    make_grid();
    for (unsigned int cycle = 0; cycle <= n_refinements; cycle++){
      std::cout << "Refinement cycle " << cycle << std::endl;
      setup_discrete_level_set();
      setup_discrete_level_set_2();
      std::cout << "Classifying cells" << std::endl;
      mesh_classifier.reclassify();
      mesh_classifier_2.reclassify();
      distribute_dofs();
      //initialize_matrices();
      //assemble_system();
      //solve();
      if (cycle == n_refinements)
        output_results();
      double interface = 0.0;
      double inside = 0.0;
      //double interface_2 = 0.0;
      double inside_2 = 0.0;
      int repeat=1;
      double accumulation=0.0;
      double construction=0.0;
      double accumulation_inside=0.0;
      double construction_inside=0.0;
      for (int i=0; i<repeat; i++) {
          interface += (1.0/repeat)*compute_interface();
          inside += (1.0/repeat)*compute_inside();
          accumulation += (1.0/repeat)*accumulation_time;
          construction += (1.0/repeat)*construction_time;
          accumulation_inside += (1.0/repeat)*accumulation_time_inside;
          construction_inside += (1.0/repeat)*construction_time_inside;
        }


      const double cell_side_interface =
        triangulation.begin_active()->minimum_vertex_distance();
      const int iterations=solve_iter;
      convergence_table.add_value("Cycle", cycle);
      convergence_table.add_value("Mesh size", cell_side_interface);
      convergence_table.add_value("Cut cells", intersected_cells);
      convergence_table.evaluate_convergence_rates(
        "Cut cells", ConvergenceTable::reduction_rate_log2);
      convergence_table.add_value("CPU_interface_construction", construction);
      convergence_table.evaluate_convergence_rates(
        "CPU_interface_construction", ConvergenceTable::reduction_rate_log2);
      convergence_table.set_scientific("CPU_interface_construction", true);
      convergence_table.add_value("CPU_interface_accumulation", accumulation);
      convergence_table.evaluate_convergence_rates(
        "CPU_interface_accumulation", ConvergenceTable::reduction_rate_log2);
      convergence_table.set_scientific("CPU_interface_accumulation", true);


      convergence_table.add_value("CPU_intersect_construction", construction_inside);
      convergence_table.evaluate_convergence_rates(
        "CPU_intersect_construction", ConvergenceTable::reduction_rate_log2);
      convergence_table.set_scientific("CPU_intersect_construction", true);


      convergence_table.add_value("CPU_intersect_accumulation", accumulation_inside);
      convergence_table.set_scientific("CPU_intersect_accumulation", true);
      convergence_table.evaluate_convergence_rates(
        "CPU_intersect_accumulation", ConvergenceTable::reduction_rate_log2);


      convergence_table.add_value("Error_interface", abs((interface)-cap_area));
      convergence_table.evaluate_convergence_rates(
        "Error_interface", ConvergenceTable::reduction_rate_log2);
      convergence_table.set_scientific("Error_interface", true);


      convergence_table.add_value("Error_inside", abs((inside)-cap_volume));
      convergence_table.evaluate_convergence_rates(
        "Error_inside", ConvergenceTable::reduction_rate_log2);
      convergence_table.set_scientific("Error_inside", true);




      std::cout << std::endl;
      convergence_table.write_text(std::cout);
      std::cout << std::endl;
      triangulation.refine_global(1);
    }
  }


} // namespace StepTrace






// @sect3{The main() function}
int main()
{
  const int dim = 2;


  StepTrace::LaplaceBeltramiSolver<dim> LB_solver;
  LB_solver.run();
}