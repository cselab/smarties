!==============================================================================
!
! app_main.f90
! Part of the cart_pole_f90 example.
!
! This file contains the 'app_main' function, called by Smarties, where the
! training takes place.
!
! For clarity:
!
! C++     Interface        Fortran
! double  real(c_double)   double precision = real*8 = real(kind=8)
! bool    logical(c_bool)  logical
! int     integer(c_int)   integer
! *       type(c_ptr)      Fortran does not really like explicit pointers
!
!
! Copyright (c) 2019 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
! Distributed under the terms of the MIT license.
!
!==============================================================================

module app_main_module
  
  implicit none

  include 'mpif.h'

  private
  integer, parameter :: NUM_ACTIONS = 1
  integer, parameter :: STATE_SIZE  = 6

  ! 'app_main' is called from the outside (by the interface function
  ! 'app_main_interface' located in 'main.cpp')
  public app_main

contains

  function app_main(smarties_comm, f_mpicomm) result(result_value) &
    bind(c, name='app_main')
    ! This is the main function called from Smarties when the training starts.
  
    use, intrinsic :: iso_c_binding
    use smarties
    use cart_pole
    implicit none
  
    type(c_ptr),    intent(in), value :: smarties_comm ! this is the pointer to the Smarties communicator
    integer(c_int), intent(in), value :: f_mpicomm ! this is the MPI_COMM_WORLD handle initialized by Smarties
    !
    integer(c_int) :: result_value
    !
    type(cartPole) :: env ! define one instance of the environment class
    !                     ! in this example, the cartPole class is defined in 'cart_pole.f90'
    !
    ! definition of the parameters used by Smarties
    logical(c_bool) :: bounded
    real(c_double),  dimension(NUM_ACTIONS), target :: upper_action_bound, lower_action_bound
    logical(c_bool), dimension(STATE_SIZE),  target :: b_observable
    real(c_double),  dimension(STATE_SIZE),  target :: upper_state_bound, lower_state_bound
    real(c_double),  dimension(NUM_ACTIONS), target :: action
    logical(c_bool) :: terminated
    real(c_double), dimension(STATE_SIZE), target :: state
    real(c_double) :: reward

    integer :: rank, numProcs, mpiIerr
  

    write(*,*) 'Fortran side begins'

    ! initialize MPI ranks and check that things are working
    call mpi_comm_rank(f_mpicomm, rank, mpiIerr)
    call mpi_comm_size(f_mpicomm, numProcs, mpiIerr)
    write(*,*) 'rank #', rank, ' of ', numProcs, ' is alive in Fortran'


    ! inform Smarties about the size of the state and the number of actions it can take
    call smarties_setStateActionDims(smarties_comm, STATE_SIZE, NUM_ACTIONS)
  
    ! OPTIONAL: aciton bounds
    bounded = .true.
    upper_action_bound = (/ 10/)
    lower_action_bound = (/-10/)
    call smarties_setActionScales(smarties_comm, &
        c_loc(upper_action_bound), c_loc(lower_action_bound), &
        bounded, NUM_ACTIONS)
  
    ! OPTIONAL: hide state variables.
    ! e.g. show cosine/sine but not angle
    b_observable = (/.true., .true., .true., .false., .true., .true./)
    call smarties_setStateObservable(smarties_comm, c_loc(b_observable), STATE_SIZE)
  
    ! OPTIONAL: set space bounds
    upper_state_bound = (/ 1,  1,  1,  1,  1,  1/)
    lower_state_bound = (/-1, -1, -1, -1, -1, -1/)
    call smarties_setStateScales(smarties_comm, c_loc(upper_state_bound), c_loc(lower_state_bound), STATE_SIZE)
  
    ! train loop
    do while (.true.)
  
      ! reset environment
      call env%reset()
      
      ! send initial state to Smarties
      state = env%getState()
      call smarties_sendInitState(smarties_comm, c_loc(state), STATE_SIZE)
  
      ! simulation loop
      do while (.true.)
  
        ! get the action
        call smarties_recvAction(smarties_comm, c_loc(action), NUM_ACTIONS)
  
        ! advance the simulation
        terminated = env%advance(action)
  
        ! get the current state and reward
        state = env%getState()
        reward = env%getReward()
  
        if (terminated) then
          ! tell Smarties that this is a terminal state
          call smarties_sendTermState(smarties_comm, c_loc(state), STATE_SIZE, reward)
          exit
        else
          call smarties_sendState(smarties_comm, c_loc(state), STATE_SIZE, reward)
        end if
  
      end do ! simulation loop
  
    end do ! train loop
  
  
    result_value = 0
    write(*,*) 'Fortran side ends'

  end function app_main

end module app_main_module
