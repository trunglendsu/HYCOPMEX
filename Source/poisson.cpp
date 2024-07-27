#include "poisson.H"
#include "kn_poisson.H"

#include <AMReX_BCUtil.H>
#include <AMReX_MLMG.H>
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MultiFabUtil.H>

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_BCRec.H>
#include <AMReX_GMRES_MLMG.H>

using namespace amrex;
//--------------------------------------
//++++++++++++++++++++++++++++++++++++++
//++++++++++++++++++++++++++++++++++++++

void init_phi(MultiFab& userCtx)
{
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for( MFIter mfi(userCtx); mfi.isValid(); ++mfi )
    {
        const Box& vbx = mfi.validbox();
        auto const& ctx  = userCtx.array(mfi);

        amrex::ParallelFor(vbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            ctx(i, j, k, 1) = Real(0.0);
        });
    }
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//----- Poisson solver -- main one ---------------------------
//------------------------------------------------------------
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

void poisson_advance( MultiFab& poisson_sol,
                      MultiFab& poisson_rhs,
                      Geometry const& geom,
                      BoxArray const& grids,
                      DistributionMapping const& dmap,
                      Vector<BCRec> const& bc )
{
/*
 * We use an MLABecLaplacian operator:
 * (ascalar*acoef - bscalar div bcoef grad) phi = RHS
 * for an implicit discretization of the heat equation
 * (I - div dt grad) phi^{n+1} = phi^n
*/

    // initialize valid and ghost region to zero
    poisson_sol.setVal(0.0);

    // assorment of solver and parallization options and parameters
    // see AMReX_MLLinOp.H for the defaults, accessors, and mutators
    LPInfo info;

    // Implicit solve using MLABecLaplacian class
    MLABecLaplacian mlabec({geom}, {grids}, {dmap}, info);

    // order of stencil
    int linop_maxorder = 2;
    mlabec.setMaxOrder(linop_maxorder);

    // build array of boundary conditions needed by MLABecLaplacian
    // see Src/Boundary/AMReX_LO_BCTYPES.H for supported types
    std::array<LinOpBCType,AMREX_SPACEDIM> LinOp_bc_lo;
    std::array<LinOpBCType,AMREX_SPACEDIM> LinOp_bc_hi;

    for (int n = 0; n < poisson_sol.nComp(); ++n)
    {
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
        {
            // lo-side BCs
            if (bc[n].lo(idim) == BCType::int_dir) {
                LinOp_bc_lo[idim] = LinOpBCType::Periodic;
            }
            else if (bc[n].lo(idim) == BCType::foextrap) {
                LinOp_bc_lo[idim] = LinOpBCType::Neumann;
            }
            else if (bc[n].lo(idim) == BCType::ext_dir) {
                LinOp_bc_lo[idim] = LinOpBCType::Dirichlet;
            }
            else {
                amrex::Abort("Invalid bc_lo");
            }

            // hi-side BCs
            if (bc[n].hi(idim) == BCType::int_dir) {
                LinOp_bc_hi[idim] = LinOpBCType::Periodic;
            }
            else if (bc[n].hi(idim) == BCType::foextrap) {
                LinOp_bc_hi[idim] = LinOpBCType::Neumann;
            }
            else if (bc[n].hi(idim) == BCType::ext_dir) {
                LinOp_bc_hi[idim] = LinOpBCType::Dirichlet;
            }
            else {
                amrex::Abort("Invalid bc_hi");
            }
        }
    }

    // tell the solver what the domain boundary conditions are
    mlabec.setDomainBC(LinOp_bc_lo, LinOp_bc_hi);

    // set the boundary conditions
    // This loads the value of the Neumman boundary condition at the ghost cells
    // Ghost cell stores the value of the BCs for the Neumann
    mlabec.setLevelBC(0, &poisson_sol);

    // scaling factors
    Real ascalar = 0.0;
    Real bscalar = -1.0;
    mlabec.setScalars(ascalar, bscalar);

    // Set up coefficient matrices
    MultiFab acoef(grids, dmap, 1, 0);

    // fill in the acoef MultiFab and load this into the solver
    acoef.setVal(0.0);
    mlabec.setACoeffs(0, acoef);
    // We need to check this ? What is the coefficent for b for ??
    // bcoef.setVal(1.0);
    // mlabec.setBCoeffs(0, bcoef);


    // bcoef lives on faces so we make an array of face-centered MultiFabs
    // then we will in face_bcoef MultiFabs and load them into the solver.
    std::array<MultiFab,AMREX_SPACEDIM> face_bcoef;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        const BoxArray& ba = amrex::convert(acoef.boxArray(),
                                            IntVect::TheDimensionVector(idim));
        face_bcoef[idim].define(ba, acoef.DistributionMap(), 1, 0);
        face_bcoef[idim].setVal(1.0);
    }
    mlabec.setBCoeffs(0, amrex::GetArrOfConstPtrs(face_bcoef));


    // build an MLMG solver
    MLMG mlmg(mlabec);

    // set solver parameters
    int max_iter = 1000;
    mlmg.setMaxIter(max_iter);

    int max_fmg_iter = 0;
    mlmg.setMaxFmgIter(max_fmg_iter);

    int verbose = 2;
    mlmg.setVerbose(verbose);

    int bottom_verbose = 0;
    mlmg.setBottomVerbose(bottom_verbose);
    // relative and absolute tolerances for linear solve
    const Real tol_rel = 1.0e-12;
    const Real tol_abs = 0.0;

    // Solve linear system
    mlmg.solve({&poisson_sol}, {&poisson_rhs}, tol_rel, tol_abs);
}

//++++++++++++++++++++++++++++++++++++++++++++
//-- Update the pressure and velocity field---
//-- After the projection step ---------------
//++++++++++++++++++++++++++++++++++++++++++++
void update_solution( Array<MultiFab, AMREX_SPACEDIM>& array_grad_p,
                      Array<MultiFab, AMREX_SPACEDIM>& array_grad_phi,
                      MultiFab& fluxPrsGrad,
                      MultiFab& cc_grad_phi,
                      MultiFab& poisson_rhs,
                      MultiFab& userCtx,
                      MultiFab& velCart,
                      Array<MultiFab, AMREX_SPACEDIM>& velCont,
                      Array<MultiFab, AMREX_SPACEDIM>& velContPrev,
                      Array<MultiFab, AMREX_SPACEDIM>& velContDiff,
                      Geometry const& geom,
                      Real const& dt,
                      const int& Nghost,
                      Vector<int> const& phy_bc_lo,
                      Vector<int> const& phy_bc_hi,
                      const int& n_cell,
                      Real const& ren )
{
    /*
    gradient_calc_approach1(fluxPrsGrad, cc_grad_phi, userCtx, geom, Nghost, phy_bc_lo, phy_bc_hi, n_cell);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(array_grad_phi[0]); mfi.isValid(); ++mfi )
    {
        const Box& xbx = mfi.tilebox(IntVect(AMREX_D_DECL(1,0,0)));
        const Box& ybx = mfi.tilebox(IntVect(AMREX_D_DECL(0,1,0)));
#if (AMREX_SPACEDIM > 2)

        const Box& zbx = mfi.tilebox(IntVect(AMREX_D_DECL(0,0,1)));
#endif
        auto const& grad_phi_x = array_grad_phi[0].array(mfi);
        auto const& grad_phi_y = array_grad_phi[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const& grad_phi_z = array_grad_phi[2].array(mfi);
#endif

        auto const& grad_phi = cc_grad_phi.array(mfi);

        amrex::ParallelFor(xbx,
                           [=] AMREX_GPU_DEVICE (int i, int j, int k){ 
            grad_phi_x(i, j, k) = amrex::Real(0.5)*( grad_phi(i-1, j, k, 0) + grad_phi(i, j, k, 0) );
        });

        amrex::ParallelFor(ybx,
                           [=] AMREX_GPU_DEVICE (int i, int j, int k){ 
            grad_phi_y(i, j, k) = amrex::Real(0.5)*( grad_phi(i, j-1, k, 1) + grad_phi(i, j, k, 1) );
        });
#if (AMREX_SPACEDIM > 2)
        amrex::ParallelFor(zbx,
                           [=] AMREX_GPU_DEVICE (int i, int j, int k){ 
            grad_phi_z(i, j, k) = amrex::Real(0.5)*( grad_phi(i, j, k-1, 2) + grad_phi(i, j, k, 2) );
        });
#endif
    }
    */
    gradient_calc_approach2(array_grad_p, array_grad_phi, userCtx, geom, Nghost, phy_bc_lo, phy_bc_hi, n_cell);
    //==============================================
    //----- Update the pressure field --------------
    //----------------------------------------------
    for ( MFIter mfi(userCtx); mfi.isValid(); ++mfi )
    {
        const Box& vbx = mfi.validbox();
        auto const& ctx = userCtx.array(mfi);
        auto const& grad_u = poisson_rhs.array(mfi);

        amrex::ParallelFor(vbx,
                           [=] AMREX_GPU_DEVICE (int i, int j, int k){
            //Update the pressure field
            ctx(i, j, k, 0) = ctx(i, j, k, 0) + ctx(i, j, k, 1);// - ( dt * grad_u(i, j, k, 0) )/Real(1.5)/ren;
        });
    } // End of the loop for boxes

    //==============================================
    //----- Update the contravariant velocity ------
    //----------------------------------------------
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(velCont[0]); mfi.isValid(); ++mfi )
    {
        const Box& xbx = mfi.tilebox(IntVect(AMREX_D_DECL(1,0,0)));
        const Box& ybx = mfi.tilebox(IntVect(AMREX_D_DECL(0,1,0)));
#if (AMREX_SPACEDIM > 2)
        const Box& zbx = mfi.tilebox(IntVect(AMREX_D_DECL(0,0,1)));
#endif

        auto const& xcont = velCont[0].array(mfi);
        auto const& ycont = velCont[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const& zcont = velCont[2].array(mfi);
#endif

        auto const& grad_phi_x = array_grad_phi[0].array(mfi);
        auto const& grad_phi_y = array_grad_phi[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const& grad_phi_z = array_grad_phi[2].array(mfi);
#endif
        amrex::ParallelFor(xbx,
                           [=] AMREX_GPU_DEVICE (int i, int j, int k){
            xcont(i, j, k) = xcont(i, j, k) - ( grad_phi_x(i, j, k) * dt / Real(1.5) );
        });

        amrex::ParallelFor(ybx,
                           [=] AMREX_GPU_DEVICE (int i, int j, int k){
            ycont(i, j, k) = ycont(i, j, k) - ( grad_phi_y(i, j, k) * dt / Real(1.5) );
        });
#if (AMREX_SPACEDIM > 2)
        amrex::ParallelFor(zbx,
                           [=] AMREX_GPU_DEVICE (int i, int j, int k){
            zcont(i, j, k) = zcont(i, j, k) - ( grad_phi_z(i, j, k) * dt / Real(1.5) );
        });
#endif
    } // End of the loop for boxes
    // Update velContDiff and velContPrev from the velCont solutions
    for (int dir=0; dir<AMREX_SPACEDIM; ++dir) {
        MultiFab::Subtract(velContPrev[dir], velCont[dir], 0, 0, 1, 0);
        MultiFab::Copy(velContDiff[dir], velContPrev[dir], 0, 0, 1, 0);
        MultiFab::Copy(velContPrev[dir], velCont[dir], 0, 0, 1, 0);
    }
    // Update velCart from the velCont solutions
    cont2cart(velCart, velCont, geom, Nghost, phy_bc_lo, phy_bc_hi, n_cell);
}


//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//----- Poisson using GMRES solver -- main one ---------------------------
//------------------------------------------------------------
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

void poisson_advance_GMRES( MultiFab& poisson_sol,
                      MultiFab& poisson_rhs,
                      Geometry const& geom,
                      BoxArray const& grids,
                      DistributionMapping const& dmap,
                      Vector<BCRec> const& bc )
{
/*
 * We use an MLABecLaplacian operator:
 * (ascalar*acoef - bscalar div bcoef grad) phi = RHS
 * for an implicit discretization of the heat equation
 * (I - div dt grad) phi^{n+1} = phi^n
*/

    // initialize valid and ghost region to zero
    poisson_sol.setVal(0.0);

    // assorment of solver and parallization options and parameters
    // see AMReX_MLLinOp.H for the defaults, accessors, and mutators
    LPInfo info;

    // Implicit solve using MLABecLaplacian class
    MLABecLaplacian mlabec({geom}, {grids}, {dmap}, info);

    // order of stencil
    int linop_maxorder = 2;
    mlabec.setMaxOrder(linop_maxorder);

    // build array of boundary conditions needed by MLABecLaplacian
    // see Src/Boundary/AMReX_LO_BCTYPES.H for supported types
    std::array<LinOpBCType,AMREX_SPACEDIM> LinOp_bc_lo;
    std::array<LinOpBCType,AMREX_SPACEDIM> LinOp_bc_hi;

    for (int n = 0; n < poisson_sol.nComp(); ++n)
    {
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
        {
            // lo-side BCs
            if (bc[n].lo(idim) == BCType::int_dir) {
                LinOp_bc_lo[idim] = LinOpBCType::Periodic;
            }
            else if (bc[n].lo(idim) == BCType::foextrap) {
                LinOp_bc_lo[idim] = LinOpBCType::Neumann;
            }
            else if (bc[n].lo(idim) == BCType::ext_dir) {
                LinOp_bc_lo[idim] = LinOpBCType::Dirichlet;
            }
            else {
                amrex::Abort("Invalid bc_lo");
            }

            // hi-side BCs
            if (bc[n].hi(idim) == BCType::int_dir) {
                LinOp_bc_hi[idim] = LinOpBCType::Periodic;
            }
            else if (bc[n].hi(idim) == BCType::foextrap) {
                LinOp_bc_hi[idim] = LinOpBCType::Neumann;
            }
            else if (bc[n].hi(idim) == BCType::ext_dir) {
                LinOp_bc_hi[idim] = LinOpBCType::Dirichlet;
            }
            else {
                amrex::Abort("Invalid bc_hi");
            }
        }
    }

    // tell the solver what the domain boundary conditions are
    mlabec.setDomainBC(LinOp_bc_lo, LinOp_bc_hi);

    // set the boundary conditions
    // This loads the value of the Neumman boundary condition at the ghost cells
    // Ghost cell stores the value of the BCs for the Neumann
    mlabec.setLevelBC(0, &poisson_sol);

    // scaling factors
    Real ascalar = 0.0;
    Real bscalar = -1.0;
    mlabec.setScalars(ascalar, bscalar);

    // Set up coefficient matrices
    MultiFab acoef(grids, dmap, 1, 0);

    // fill in the acoef MultiFab and load this into the solver
    acoef.setVal(0.0);
    mlabec.setACoeffs(0, acoef);
    // We need to check this ? What is the coefficent for b for ??
    // bcoef.setVal(1.0);
    // mlabec.setBCoeffs(0, bcoef);


    // bcoef lives on faces so we make an array of face-centered MultiFabs
    // then we will in face_bcoef MultiFabs and load them into the solver.
    std::array<MultiFab,AMREX_SPACEDIM> face_bcoef;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        const BoxArray& ba = amrex::convert(acoef.boxArray(),
                                            IntVect::TheDimensionVector(idim));
        face_bcoef[idim].define(ba, acoef.DistributionMap(), 1, 0);
        face_bcoef[idim].setVal(1.0);
    }
    mlabec.setBCoeffs(0, amrex::GetArrOfConstPtrs(face_bcoef));


    // build an MLMG solver
    MLMG mlmg(mlabec);

    // set solver parameters
    int max_iter = 1000;
    mlmg.setMaxIter(max_iter);

    int max_fmg_iter = 0;
    mlmg.setMaxFmgIter(max_fmg_iter);

    int verbose = 2;
    mlmg.setVerbose(verbose);

    int bottom_verbose = 0;
    mlmg.setBottomVerbose(bottom_verbose);
    // relative and absolute tolerances for linear solve
    const Real tol_rel = 1.0e-12;
    const Real tol_abs = 0.0;

    // Solve linear system
    // mlmg.solve({&poisson_sol}, {&poisson_rhs}, tol_rel, tol_abs);

    // Setup the GMRES solver
    GMRESMLMGT<MultiFab> my_solver(mlmg);
    my_solver.setVerbose(10);
    my_solver.setMaxIters(20);
    // my_solver.usePrecond(false);

    my_solver.solve(poisson_sol, poisson_rhs, tol_rel, tol_abs);

    

}
