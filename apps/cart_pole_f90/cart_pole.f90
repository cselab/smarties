!==============================================================================
!
! cart_pole.f90
! Part of the cart_pole_f90 example.
!
! This file contains the environment (cart pole) definitions in Fortran.
! The module 'cart_pole' defines the environment and the simulation
! parameters.
!
!
! Copyright (c) 2019 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
! Distributed under the terms of the MIT license.
!
!==============================================================================

module cart_pole

  implicit none

  private
  double precision, parameter :: pi = 3.1415926535897931d0
  logical,          parameter :: SWINGUP=.false.
  double precision, parameter :: mp = 0.1
  double precision, parameter :: mc = 1.
  double precision, parameter :: l  = 0.5
  double precision, parameter :: g  = 9.81
  integer,          parameter :: STATE_SIZE = 6


  type, public :: cartPole
    ! Main class containing the simulation data and procedures
    double precision :: dt = 4e-4
    integer :: nsteps = 50
    integer :: step = 0
    double precision, dimension(4) :: u
    double precision :: F=0, t=0
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
    double precision, dimension(STATE_SIZE) :: state
    !
    state(1:4) = this%u
    state(5)   = cos(this%u(3))
    state(6)   = sin(this%u(3))
  end function getState


  function getReward(this) result(reward) ! assign a reward associated with the current state
    class(cartPole), intent(in) :: this
    double precision :: angle
    double precision :: reward
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
    double precision, intent(in) :: t0, dt, F
    double precision, dimension(4), intent(in) :: u0
    double precision, dimension(4) :: u
    integer :: i
    double precision :: t
    double precision, dimension(4) :: w=0
    integer, parameter :: s = 6
    double precision, dimension(6), parameter :: a = (/0.000000000000, -0.737101392796, -1.634740794341, &
        -0.744739003780, -1.469897351522, -2.813971388035/)
    double precision, dimension(6), parameter :: b = (/0.032918605146,  0.823256998200,  0.381530948900, &
         0.200092213184,  1.718581042715,  0.270000000000/)
    double precision, dimension(6), parameter :: c = (/0.000000000000,  0.032918605146,  0.249351723343, &
         0.466911705055,  0.582030414044,  0.847252983783/)
    !
    u = u0
    do i = 1, s
      t = t0 + dt*c(i)
      w = w*a(i) + Diff(u, F)*dt
      u = u + w*b(i)
    end do
  end function rk46_nl


  function Diff(u, F) result(res) ! part of time discretization
    double precision, dimension(4), intent(in) :: u
    double precision, intent(in) :: F
    double precision, dimension(4) :: res
    !
    double precision :: cosy, siny, w
    !
    double precision :: fac1, fac2, totMass, F1
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

end module cart_pole
