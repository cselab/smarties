!==============================================================================
!
! smarties.f90
!
! This module, usable by any Fortran code with 'use smarties' in the preamble,
! defines the interfaces to the C/C++ functions, which are located in
! 'source/smarties_extern.cpp'.
!
! Interfaces can also be created for functions that take different kinds of
! variables (with the same name) as arguments.  Two interfaces of this kind are
! implemented for 'smarties_set_action_scales' and
! 'smarties_set_action_options'.  Take a look at these as examples if you need
! to create more.
!
! *****************************************************************************
! ********** There should be no need to modify any of the functions ***********
! **** If any of the following is edited, the corresponding C/C++ function ****
! *** located in 'source/smarties_extern.cpp' should be edited accordingly ****
! *****************************************************************************
!
!
! Copyright (c) 2019 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
! Distributed under the terms of the MIT license.
!
!==============================================================================

module smarties

  implicit none

  interface
    subroutine smarties_sendInitState( &
        ptr2comm, state, state_dim, agentID) &
        bind(c, name='smarties_sendInitState')
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),    intent(in), value :: ptr2comm
      type(c_ptr),    intent(in), value :: state
      integer(c_int), intent(in), value :: state_dim
      integer(c_int), intent(in), optional :: agentID
    end subroutine smarties_sendInitState
  end interface

  interface
    subroutine smarties_sendState( &
        ptr2comm, state, state_dim, reward, agentID) &
        bind(c, name='smarties_sendState')
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),    intent(in), value :: ptr2comm
      type(c_ptr),    intent(in), value :: state
      integer(c_int), intent(in), value :: state_dim
      real(c_double), intent(in), value :: reward
      integer(c_int), intent(in), optional :: agentID
    end subroutine smarties_sendState
  end interface

  interface
    subroutine smarties_sendTermState( &
        ptr2comm, state, state_dim, reward, agentID) &
        bind(c, name='smarties_sendTermState')
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),    intent(in), value :: ptr2comm
      type(c_ptr),    intent(in), value :: state
      integer(c_int), intent(in), value :: state_dim
      real(c_double), intent(in), value :: reward
      integer(c_int), intent(in), optional :: agentID
    end subroutine smarties_sendTermState
  end interface

  interface
    subroutine smarties_sendLastState( &
        ptr2comm, state, state_dim, reward, agentID) &
        bind(c, name='smarties_sendLastState')
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),    intent(in), value :: ptr2comm
      type(c_ptr),    intent(in), value :: state
      integer(c_int), intent(in), value :: state_dim
      real(c_double), intent(in), value :: reward
      integer(c_int), intent(in), optional :: agentID
    end subroutine smarties_sendLastState
  end interface

  interface
    subroutine smarties_recvAction( &
        ptr2comm, action, action_dim, agentID) &
        bind(c, name='smarties_recvAction')
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),    intent(in), value :: ptr2comm
      type(c_ptr),    intent(in), value :: action
      integer(c_int), intent(in), value :: action_dim
      integer(c_int), intent(in), optional :: agentID
    end subroutine smarties_recvAction
  end interface

  interface
    subroutine smarties_setNumAgents( &
        ptr2comm, num_agents) &
        bind(c, name='smarties_setNumAgents')
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),    intent(in), value :: ptr2comm
      integer(c_int), intent(in), value :: num_agents
    end subroutine smarties_setNumAgents
  end interface

  interface
    subroutine smarties_setStateActionDims( &
        ptr2comm, state_dim, action_dim, agentID) &
        bind(c, name='smarties_setStateActionDims')
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),    intent(in), value :: ptr2comm
      integer(c_int), intent(in), value :: state_dim
      integer(c_int), intent(in), value :: action_dim
      integer(c_int), intent(in), optional :: agentID
    end subroutine smarties_setStateActionDims
  end interface


  interface
    subroutine smarties_setActionScales( &
        ptr2comm, upper_act_bound, lower_act_bound, &
        bounded, action_dim, agentID) &
        bind(c, name='smarties_setActionScales')
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),     intent(in), value :: ptr2comm
      type(c_ptr),     intent(in), value :: upper_act_bound
      type(c_ptr),     intent(in), value :: lower_act_bound
      logical(c_bool), intent(in), value :: bounded
      integer(c_int),  intent(in), value :: action_dim
      integer(c_int),  intent(in), optional :: agentID
    end subroutine smarties_setActionScales
  end interface

  interface
    subroutine smarties_setActionScalesBounds( &
        ptr2comm, upper_act_bound, lower_act_bound, &
        bounded, action_dim, agentID) &
        bind(c, name='smarties_setActionScalesBounds')
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),     intent(in), value :: ptr2comm
      type(c_ptr),     intent(in), value :: upper_act_bound
      type(c_ptr),     intent(in), value :: lower_act_bound
      type(c_ptr),     intent(in), value :: bounded
      integer(c_int),  intent(in), value :: action_dim
      integer(c_int),  intent(in), optional :: agentID
    end subroutine smarties_setActionScalesBounds
  end interface


  interface
    subroutine smarties_setActionOptions( &
        ptr2comm, num_options, agentID) &
        bind(c, name='smarties_setActionOptions')
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),     intent(in), value :: ptr2comm
      integer(c_int),  intent(in), value :: num_options
      integer(c_int),  intent(in), optional :: agentID
    end subroutine smarties_setActionOptions
  end interface

  interface
    subroutine smarties_setActionOptionsPerDim( &
        ptr2comm, num_options, action_dim, agentID) &
        bind(c, name='smarties_setActionOptionsPerDim')
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),     intent(in), value :: ptr2comm
      type(c_ptr),     intent(in), value :: num_options
      integer(c_int),  intent(in), value :: action_dim
      integer(c_int),  intent(in), optional :: agentID
    end subroutine smarties_setActionOptionsPerDim
  end interface

  interface
    subroutine smarties_setStateObservable( &
        ptr2comm, b_observable, state_dim, agentID) &
        bind(c, name='smarties_setStateObservable')
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),    intent(in), value :: ptr2comm
      type(c_ptr),    intent(in), value :: b_observable
      integer(c_int), intent(in), value :: state_dim
      integer(c_int), intent(in), optional :: agentID
    end subroutine smarties_setStateObservable
  end interface

  interface
    subroutine smarties_setStateScales( &
        ptr2comm, upper_state_bound, lower_state_bound, state_dim, agentID) &
        bind(c, name='smarties_setStateScales')
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),    intent(in), value :: ptr2comm
      type(c_ptr),    intent(in), value :: upper_state_bound
      type(c_ptr),    intent(in), value :: lower_state_bound
      integer(c_int), intent(in), value :: state_dim
      integer(c_int), intent(in), optional :: agentID
    end subroutine smarties_setStateScales
  end interface

  interface
    subroutine smarties_setIsPartiallyObservable(ptr2comm, agentID) &
        bind(c, name='smarties_setIsPartiallyObservable')
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),    intent(in), value :: ptr2comm
      integer(c_int), intent(in), optional :: agentID
    end subroutine smarties_setIsPartiallyObservable
  end interface

  interface
    subroutine smarties_finalizeProblemDescription(ptr2comm) &
        bind(c, name='smarties_finalizeProblemDescription')
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),    intent(in), value :: ptr2comm
    end subroutine smarties_finalizeProblemDescription
  end interface

  interface
    subroutine smarties_envHasDistributedAgents(ptr2comm) &
        bind(c, name='smarties_envHasDistributedAgents')
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),    intent(in), value :: ptr2comm
    end subroutine smarties_envHasDistributedAgents
  end interface

  interface
    subroutine smarties_agentsDefineDifferentMDP(ptr2comm) &
        bind(c, name='smarties_agentsDefineDifferentMDP')
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),    intent(in), value :: ptr2comm
    end subroutine smarties_agentsDefineDifferentMDP
  end interface

  interface
    subroutine smarties_disableDataTrackingForAgents( &
        ptr2comm, agentStart, agentEnd) &
        bind(c, name='smarties_disableDataTrackingForAgents')
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),    intent(in), value :: ptr2comm
      integer(c_int), intent(in), value :: agentStart
      integer(c_int), intent(in), value :: agentEnd
    end subroutine smarties_disableDataTrackingForAgents
  end interface

  interface
    subroutine smarties_setPreprocessingConv2d( &
        ptr2comm, input_width, input_height, input_features, &
        kernels_num, filters_size, stride, agentID) &
        bind(c, name='smarties_setPreprocessingConv2d')
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),    intent(in), value :: ptr2comm
      integer(c_int), intent(in), value :: input_width
      integer(c_int), intent(in), value :: input_height
      integer(c_int), intent(in), value :: input_features
      integer(c_int), intent(in), value :: kernels_num
      integer(c_int), intent(in), value :: filters_size
      integer(c_int), intent(in), value :: stride
      integer(c_int), intent(in), optional :: agentID
    end subroutine smarties_setPreprocessingConv2d
  end interface

  interface
    subroutine smarties_setNumAppendedPastObservations( &
        ptr2comm, n_appended, agentID) &
        bind(c, name='smarties_setNumAppendedPastObservations')
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),    intent(in), value :: ptr2comm
      integer(c_int), intent(in), value :: n_appended
      integer(c_int), intent(in), optional :: agentID
    end subroutine smarties_setNumAppendedPastObservations
  end interface

  interface
    subroutine smarties_getUniformRandom( &
        ptr2comm, range_begin, range_end, sampled) &
        bind(c, name='smarties_getUniformRandom')
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),    intent(in),  value    :: ptr2comm
      real(c_double), intent(in),  optional :: range_begin
      real(c_double), intent(in),  optional :: range_end
      real(c_double), intent(out), optional :: sampled(*)
    end subroutine smarties_getUniformRandom
  end interface

  interface
    subroutine smarties_getNormalRandom( &
        ptr2comm, mean, stdev, sampled) &
        bind(c, name='smarties_getNormalRandom')
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),    intent(in),  value    :: ptr2comm
      real(c_double), intent(in),  optional :: mean
      real(c_double), intent(in),  optional :: stdev
      real(c_double), intent(out), optional :: sampled(*)
    end subroutine smarties_getNormalRandom
  end interface

end module smarties
