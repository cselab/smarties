!
!  cart_pole.f90
!  cart-pole
!
!  Created by Jacopo Canton on 31/01/19
!  Copyright (c) 2019 Jacopo Canton. All rights reserved.
!
!  This file contains the cart pole test case for Smarties in Fortran.
!   - The first module, `class_cartPole`, defines the environment and the
!   simulation parameters.
!   - The second module, `fortran_smarties`, contains the interface to the C++
!   code as well as the main function `fortran_app_main` called by Smarties.
!
!==============================================================================


!==============================================================================

module fortran_smarties
  ! Module interfacing with Smarties
  
  implicit none

  include 'mpif.h'

  private
  integer, parameter :: NUM_ACTIONS = 1
  integer, parameter :: STATE_SIZE  = 6

  ! Just to be clear on our intention for this procedure to be called from
  ! outside the module.
  public fortran_app_main

contains

  function fortran_app_main(rlcomm, f_mpicomm) result(result_value) &
    bind(c, name='fortran_app_main')
    ! This is the main function called from Smarties when the training starts.
  
    use, intrinsic :: iso_c_binding
    use class_cartPole
    implicit none
  
    type(c_ptr), intent(in), value :: rlcomm ! this is the pointer to the Smarties communicator
    integer,     intent(in), value :: f_mpicomm ! this is the MPI_COMM_WORLD handle initialized by Smarties
    !
    integer :: result_value
    !
    type(cartPole) :: env ! define one instance of the environment class
    !
    ! definition of the parameters used with Smarties
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
    call rlcomm_update_state_action_dims(rlcomm, STATE_SIZE, NUM_ACTIONS)
  
    ! OPTIONAL: aciton bounds
    bounded = .true.
    upper_action_bound = (/ 10/)
    lower_action_bound = (/-10/)
    call rlcomm_set_action_scales(rlcomm, c_loc(upper_action_bound), c_loc(lower_action_bound), NUM_ACTIONS, bounded)
  
    ! OPTIONAL: hide state variables.
    ! e.g. show cosine/sine but not angle
    b_observable = (/.true., .true., .true., .false., .true., .true./)
    call rlcomm_set_state_observable(rlcomm, c_loc(b_observable), STATE_SIZE)
  
    ! OPTIONAL: set space bounds
    upper_state_bound = (/ 1,  1,  1,  1,  1,  1/)
    lower_state_bound = (/-1, -1, -1, -1, -1, -1/)
    call rlcomm_set_state_scales(rlcomm, c_loc(upper_state_bound), c_loc(lower_state_bound), STATE_SIZE)
  
    ! train loop
    do while (.true.)
  
      ! reset environment
      call env%reset()
      
      ! send initial state to Smarties
      state = env%getState()
      call rlcomm_sendInitState(rlcomm, c_loc(state), STATE_SIZE)
  
      ! simulation loop
      do while (.true.)
  
        ! get the action
        call rlcomm_recvAction(rlcomm, c_loc(action), NUM_ACTIONS)
  
        ! advance the simulation
        terminated = env%advance(action)
  
        ! get the current state and reward
        state = env%getState()
        reward = env%getReward()
  
        if (terminated) then
          ! tell Smarties that this is a terminal state
          call rlcomm_sendTermState(rlcomm, c_loc(state), STATE_SIZE, reward)
          exit
        else
          call rlcomm_sendState(rlcomm, c_loc(state), STATE_SIZE, reward)
        end if
  
      end do ! simulation loop
  
    end do ! train loop
  
  
    result_value = 0
    write(*,*) 'Fortran side ends'

  end function fortran_app_main

end module fortran_smarties
