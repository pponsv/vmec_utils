module vh

   implicit none

contains

   subroutine genvar(fnm, th, ph, ms, ns, n_s, n_m, n_th, n_ph, typ, out)
      integer(8), intent(in) :: n_m, n_th, n_ph, n_s
      real(8), intent(in) :: fnm(n_m, n_s), th(n_th), ph(n_ph), ms(n_m), ns(n_m)
      character, intent(in) :: typ
      real(8), intent(out) :: out(n_s, n_th, n_ph)

      real(8) :: mth(n_m, n_th), nph(n_m, n_ph), arg(n_m)
      integer(8) :: i, j, k

      mth = outer(ms, th)
      nph = outer(ns, ph)
      !$OMP PARALLEL DO PRIVATE(i, j, k, arg)
      do k = 1, n_ph
         do j = 1, n_th
            if (typ == 'c') arg = cos(mth(:, j) - nph(:, k))
            if (typ == 's') arg = sin(mth(:, j) - nph(:, k))
            do i = 1, n_s
               out(i, j, k) = sum(fnm(:, i)*arg)
            end do
         end do
      end do
      !$OMP END PARALLEL DO
   end subroutine genvar

   subroutine genvar_modi(fnm, th, ph, ms, ns, modi, n_s, n_m, n_th, n_ph, typ, out)
      integer(8), intent(in) :: n_m, n_th, n_ph, n_s
      real(8), intent(in) :: fnm(n_m, n_s), th(n_th), ph(n_ph), ms(n_m), ns(n_m), modi(n_m)
      character, intent(in) :: typ
      real(8), intent(out) :: out(n_s, n_th, n_ph)

      real(8) :: mth(n_m, n_th), nph(n_m, n_ph), arg(n_m)
      integer(8) :: i, j, k

      mth = outer(ms, th)
      nph = outer(ns, ph)
      !$OMP PARALLEL DO PRIVATE(i, j, k, arg)
      do k = 1, n_ph
         do j = 1, n_th
            if (typ == 'c') arg = modi*cos(mth(:, j) - nph(:, k))
            if (typ == 's') arg = modi*sin(mth(:, j) - nph(:, k))
            do i = 1, n_s
               out(i, j, k) = sum(fnm(:, i)*arg)
            end do
         end do
      end do
      !$OMP END PARALLEL DO
   end subroutine genvar_modi

   function dfnm_ds(fnm, s, n_m, n_s) result(dfnm)
      real(8), intent(in) :: s(n_s), fnm(n_m, n_s)
      integer(8), intent(in) :: n_m, n_s
      real(8) :: dfnm(n_m, n_s)
      integer(8) :: i
      !  Finite differences coefficients from:
      !  https://web.media.mit.edu/~crtaylor/calculator.html
      !$OMP PARALLEL DO PRIVATE(i)
      do i = 1, n_s
         if (i == 1) then
            ! dfnm(:, i) = (fnm(:, i + 1) - fnm(:, i))/((s(i + 1) - s(i))) ! First order forward
            dfnm(:, i) = (4*fnm(:, i + 1) - 3*fnm(:, i) - fnm(:, i + 2))/(2*(s(i + 1) - s(i))) ! Second order forward
         else if (i == n_s) then
            ! dfnm(:, i) = (fnm(:, i) - fnm(:, i - 1))/((s(i) - s(i - 1))) ! First order backwards
            dfnm(:, i) = (fnm(:,i-2) + 3*fnm(:, i) - 4*fnm(:, i - 1))/(2*(s(i) - s(i - 1))) ! Second order backwards
         else
            dfnm(:, i) = (fnm(:, i + 1) - fnm(:, i - 1))/((s(i + 1) - s(i - 1)))
         end if
      end do
      !$OMP END PARALLEL DO
   end function dfnm_ds

   subroutine dgen_ds(fnm, s, th, ph, ms, ns, n_s, n_m, n_th, n_ph, typ, out)
      integer(8), intent(in) :: n_m, n_th, n_ph, n_s
      real(8), intent(in) :: fnm(n_m, n_s), th(n_th), ph(n_ph), ms(n_m), ns(n_m), s(n_s)
      real(8), intent(out) :: out(n_s, n_th, n_ph)
      character, intent(in) :: typ
      real(8) :: dfnm(n_m, n_s)

      dfnm = dfnm_ds(fnm, s, n_m, n_s)

      call genvar(dfnm, th, ph, ms, ns, n_s, n_m, n_th, n_ph, typ, out)

   end subroutine dgen_ds

   function outer(a, b) result(c)
      real(8) :: a(:), b(:)
      real(8) :: c(size(a), size(b))

      integer(8) :: i, j
      do i = 1, size(b)
         do j = 1, size(a)
            c(j, i) = a(j)*b(i)
         end do
      end do
   end function outer

end module vh

! subroutine cosvar(fnm, th, ph, ms, ns, n_s, n_m, n_th, n_ph, out)
!    integer(8), intent(in) :: n_m, n_th, n_ph, n_s
!    real(8), intent(in) :: fnm(n_m, n_s), th(n_th), ph(n_ph), ms(n_m), ns(n_m)
!    real(8), intent(out) :: out(n_s, n_th, n_ph)

!    real(8) :: mth(n_m, n_th), nph(n_m, n_ph), arg(n_m)
!    integer(8) :: i, j, k

!    mth = outer(ms, th)
!    nph = outer(ns, ph)
!    !$OMP PARALLEL DO PRIVATE(i, j, k, arg)
!    do k = 1, n_ph
!       do j = 1, n_th
!          arg = cos(mth(:, j) - nph(:, k))
!          do i = 1, n_s
!             out(i, j, k) = sum(fnm(:, i)*arg)
!          end do
!       end do
!    end do
!    !$OMP END PARALLEL DO
! end subroutine cosvar

! subroutine sinvar(fnm, th, ph, ms, ns, n_s, n_m, n_th, n_ph, out)
!    integer(8), intent(in) :: n_m, n_th, n_ph, n_s
!    real(8), intent(in) :: fnm(n_m, n_s), th(n_th), ph(n_ph), ms(n_m), ns(n_m)
!    real(8), intent(out) :: out(n_s, n_th, n_ph)

!    real(8) :: mth(n_m, n_th), nph(n_m, n_ph), arg(n_m)
!    integer(8) :: i, j, k

!    mth = outer(ms, th)
!    nph = outer(ns, ph)
!    !$OMP PARALLEL DO PRIVATE(i, j, k, arg)
!    do k = 1, n_ph
!       do j = 1, n_th
!          arg = sin(mth(:, j) - nph(:, k))
!          do i = 1, n_s
!             out(i, j, k) = sum(fnm(:, i)*arg)
!          end do
!       end do
!    end do
!    !$OMP END PARALLEL DO
! end subroutine sinvar

! subroutine dsin_ds(fnm, s, th, ph, ms, ns, n_s, n_m, n_th, n_ph, out)
!    integer(8), intent(in) :: n_m, n_th, n_ph, n_s
!    real(8), intent(in) :: fnm(n_m, n_s), th(n_th), ph(n_ph), ms(n_m), ns(n_m), s(n_s)
!    real(8), intent(out) :: out(n_s, n_th, n_ph)
!    real(8) :: dfnm(n_m, n_s)

!    dfnm = dfnm_ds(fnm, s, n_m, n_s)

!    call sinvar(dfnm, th, ph, ms, ns, n_s, n_m, n_th, n_ph, out)

! end subroutine dsin_ds
