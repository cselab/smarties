!
!  cart_pole.f90
!  cart-pole
!
!  Created by Jacopo Canton on 31/01/19
!  Copyright (c) 2019 Jacopo Canton. All rights reserved.
!
!  This file contains the cart pole test case for Smarties in Fortran.
!   - The module `class_cartPole`, defines the environment and the
!   simulation parameters.
!
!==============================================================================

module class_cartPole

  implicit none
  private
  real,    parameter :: pi = 3.1415926535897931d0
  logical, parameter :: SWINGUP=.false.
  real,    parameter :: mp = 0.1
  real,    parameter :: mc = 1.
  real,    parameter :: l  = 0.5
  real,    parameter :: g  = 9.81
  integer, parameter :: STATE_SIZE = 6


  type, public :: cartPole
    ! Main class containing the simulation data and procedures
    real :: dt = 4e-4
    integer :: nsteps = 50
    integer :: step = 0
    real, dimension(4) :: u
    real :: F=0, t=0
    !
    contains
      procedure :: reset     => reset
      procedure :: is_over   => is_over
      procedure :: advance   => advance
      procedure :: getState  => getState
      procedure :: getReward => getReward
  end type cartPole

  
contains
!------------------------------------------------------------------------------
! Definition of the procedures for cartPole

  subroutine reset(this) ! reset the simulation
    class(cartPole) :: this
    ! initialize the state to a random vector
    call random_number(this%u) ! random_number is an intrinsic [0, 1)
    if (SWINGUP) then
      this%u = (this%u-0.5)*2.*1. ! [-1, 1)
    else
      this%u = (this%u-0.5)*2.*0.05 ! [-0.05, 0.05)
    end if
    this%F = 0
    this%t = 0
    this%step = 0
  end subroutine reset


  function is_over(this) result(answer) ! is the simulation over?
    class(cartPole), intent(in) :: this
    logical :: answer
    answer=.false.
    !
    if (SWINGUP) then
      if (this%step>=500 .or. abs(this%u(1))>2.4) answer = .true.
    else
      if (this%step>=500 .or. abs(this%u(1))>2.4 .or. abs(this%u(3))>pi/15 ) answer = .true.
    endif
  end function is_over


  function advance(this, action) result(terminated) ! advance the simulation in time
    use, intrinsic :: iso_c_binding
    class(cartPole) :: this
    real(c_double), dimension(:) :: action
    logical(c_bool) :: terminated
    integer :: i
    !
    this%F = action(1)
    this%step = this%step+1
    do i = 1, this%nsteps
      this%u = rk46_nl(this%t, this%dt, this%F, this%u) ! time discretization function defined below
      this%t = this%t + this%dt
      if (this%is_over()) then
        terminated = .true.
        return
      end if
    end do
    terminated = .false.
  end function advance


  function getState(this) result(state) ! get the full state (with additional values)
    class(cartPole), intent(in) :: this
    real, dimension(STATE_SIZE) :: state
    !
    state(1:4) = this%u
    state(5)   = cos(this%u(3))
    state(6)   = sin(this%u(3))
  end function getState


  function getReward(this) result(reward) ! assign a reward associated with the current state
    class(cartPole), intent(in) :: this
    real :: angle
    real :: reward
    reward = 0
    !
    if (SWINGUP) then
      angle = mod(this%u(3), 2*pi)
      if (angle<0) angle = angle + 2*pi
      if (abs(angle-pi)<pi/6) reward = 1
    else
      if (abs(this%u(3))<=pi/15 .and. abs(this%u(1))<=2.4) reward = 1
    end if
  end function getReward


  function rk46_nl(t0, dt, F, u0) result(u) ! time discretization
    real, intent(in) :: t0, dt, F
    real, dimension(4), intent(in) :: u0
    real, dimension(4) :: u
    integer :: i
    real :: t
    real, dimension(4) :: w=0
    integer, parameter :: s = 6
    real, dimension(6), parameter :: a = (/0.000000000000, -0.737101392796, -1.634740794341, &
        -0.744739003780, -1.469897351522, -2.813971388035/)
    real, dimension(6), parameter :: b = (/0.032918605146,  0.823256998200,  0.381530948900, &
         0.200092213184,  1.718581042715,  0.270000000000/)
    real, dimension(6), parameter :: c = (/0.000000000000,  0.032918605146,  0.249351723343, &
         0.466911705055,  0.582030414044,  0.847252983783/)
    !
    u = u0
    do i = 1, s
      t = t0 + dt*c(i)
      w = w*a(i) + Diff(u, t, F)*dt
      u = u + w*b(i)
    end do
  end function rk46_nl


  function Diff(u, t, F) result(res) ! part of time discretization
    real, dimension(4), intent(in) :: u
    real, intent(in) :: t, F
    real, dimension(4) :: res
    !
    real :: cosy, siny, w
    !
    real :: fac1, fac2, totMass, F1
    !
    res = 0
    cosy = cos(u(3))
    siny = sin(u(3))
    w = u(4)
    if (SWINGUP) then
      fac1 = 1./(mc + mp*siny*siny)
      fac2 = fac1/l
      res(2) = fac1*(F + mp*siny*(l*w*w + g*cosy))
      res(4) = fac2*(-F*cosy - mp*l*w*w*cosy*siny - (mc+mp)*g*siny)
    else
      totMass = mp + mc
      fac2 = l*(4./3. - mp*cosy*cosy/totMass)
      F1 = F + mp*l*w*w*siny
      res(4) = (g*siny - F1*cosy/totMass)/fac2
      res(2) = (F1 - mp*l*res(4)*cosy)/totMass
    end if
    res(1) = u(2)
    res(3) = u(4)
  end function


end module class_cartPole

!==============================================================================
