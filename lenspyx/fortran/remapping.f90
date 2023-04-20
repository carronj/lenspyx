! See python code for doc
subroutine helloworld
USE OMP_LIB

INTEGER :: thread_id

!$OMP PARALLEL PRIVATE(thread_id)

    thread_id = OMP_GET_THREAD_NUM()
    write(*, *) 'Hello from process: ', thread_id

!$OMP END PARALLEL

end subroutine

module remapping
    implicit none
    double precision, parameter :: DPI=3.141592653589793238d0, TOLAMIN=1e-10
    integer, parameter :: ITRMAX=1000

    interface d2ang
        module procedure d2ang_scal
    end interface d2ang
    interface ang2d
        module procedure ang2d_scal
    end interface ang2d

    contains

    subroutine sincd(npix, d, sind_d)
        implicit none
        double precision, intent(in) :: d(npix)
        double precision, intent(out) :: sind_d(npix)
        integer, intent(in) :: npix
        integer ipix
        do ipix=1, npix
            if (d(ipix) /= 0.d0) then
                sind_d(ipix) = dsin(d(ipix)) / d(ipix)
            else
                sind_d(ipix) = 1.d0
            end if
        end do
    end subroutine sincd

    subroutine d2ang_scal(Red, Imd, tht, phi, thtp, phip)
        ! In principle double prec. accurate and avoiding any division by zero
        ! version -1, 0, 1 for pints close to the north pole, equator or south pol
        implicit none
        double precision, intent(in) :: Red, Imd, tht, phi
        double precision, intent(out) :: thtp, phip
        double precision d, cost, costp, sind_d, sint, sintp, e_t, e_tp, e_d
        integer version
        double precision :: PI2 = DPI * 2

        d = Red * Red + Imd * Imd
        if (d > 0.0001d0) then
            write(*, *) 'too large deflection here, this should not happen', dsqrt(d), tht, phi, version
            error stop
        end if
        sind_d = 1.d0 - d / 6.d0 * (1.d0 - d / 20d0 * (1.d0 - d / 42d0))
        d = dsqrt(d)
        !version  'nearest int' 1 0 -1 for closest to north pole, equator or south pole respectively
        version = nint(1 - 2 * tht / DPI)
        if (version == 0) then
            cost = dcos(tht)
            costp = cost * dcos(d) - Red * sind_d * dsqrt(1.d0 - cost * cost)
            thtp = dacos(costp)
            phip = modulo(phi + dasin(Imd / dsqrt(1.d0 - costp * costp) * sind_d), PI2) ! ok except for absurdly large d
            return
        end if
        e_d = 2d0 * dsin(d * 0.5d0) ** 2
        sint = dsin(tht)

        if (version == 1) then
            e_t = 2d0 * dsin(tht * 0.5d0) ** 2
            e_tp = e_t + e_d - e_t * e_d +  Red * sind_d * sint
        else if (version == -1) then
            e_t = 2d0 * dcos(tht * 0.5d0) ** 2
            e_tp = e_t + e_d - e_t * e_d -  Red * sind_d * sint
        else
            write(*, *) 'invalid version parameter (must be in (-1, 0, 1))', version
            error stop
        end if
        sintp = dsqrt(dmax1(0.d0, e_tp * (2d0 - e_tp)))
        ! FIXME:
        !: the max is here to avoid machine roundoffs resulting in nans, when tht itself is machine precision to zero
        !: in practice it seems that when one tries to land exactly on the poles this reduces the precision to max 1e-11
        if (version ==  1) then
            thtp = dasin(sintp)
            phip = modulo(phi + datan2( Imd * sind_d, (1.d0 - e_d) * sint + Red * sind_d * (1.d0 - e_t)), PI2)
        else
            thtp = DPI - dasin(sintp)
            phip = modulo(phi + datan2( Imd * sind_d, (1.d0 - e_d) * sint + Red * sind_d * (e_t - 1.d0)), PI2)
        end if
        return
    end subroutine d2ang_scal

    subroutine ang2d_scal(thtp, tht, dphi, red, imd)
        implicit none
        double precision, intent(in) :: thtp, tht, dphi
        double precision, intent(out) :: red, imd
        double precision sintp, sind, norm
        sintp = dsin(thtp)
        red = dsin(thtp - tht) - 2d0 * dsin(dphi * 0.5d0) ** 2 * dcos(tht) * sintp  ! Red sind / d
        imd = dsin(dphi) * sintp  ! Imd sind / d
        sind = dsqrt(red * red + imd * imd)
        if (sind > 0d0) then
            norm = dasin(sind) / sind
            red = red * norm
            imd = imd * norm
        end if
    end subroutine ang2d_scal

    subroutine fpointing(npix, nring, red, imd, thts, phi0s, nphis, ptg, ofs, nthreads)
        ! output angles in single precision
        use OMP_LIB
        implicit none
        integer, intent(in) :: npix, nring, nthreads
        integer, intent(in) :: nphis(nring), ofs(nring)
        double precision, intent(in) :: red(npix), imd(npix), thts(nring), phi0s(nring)
        real, intent(out) :: ptg(3, npix)
        real thtp, phip, gamma
        double precision phi, dphi, sint, cost, cott, d, sind_d, e_t, e_d, e_tp, costp, sintp
        double precision :: PI2 = DPI * 2d0
        integer ir, ip, pix, version

        call OMP_SET_NUM_THREADS(nthreads)
        !$OMP PARALLEL DO DEFAULT(NONE)&
        !$OMP SHARED(PI2, thts, red, imd, ptg, nphis, phi0s, npix, nring, ofs)&
        !$OMP PRIVATE(ip, thtp, phip,gamma, e_t, sint, cost, cott, e_d, e_tp, costp, sintp, pix, phi, dphi, version, d, sind_d)
        do ir = 1, nring
            sint = dsin(thts(ir))
            cost = dcos(thts(ir))
            cott = cost / sint
            version = nint(1d0 - 2d0 * thts(ir) / DPI)
            pix = ofs(ir) + 1
            phi = phi0s(ir)
            dphi =  PI2  / nphis(ir)
            if (version == 0) then
                do ip = 1, nphis(ir)
                    d = red(pix) * red(pix) + imd(pix) * imd(pix)
                    if (d > 0d0) then
                        d = dsqrt(d)
                        sind_d = dsin(d) / d
                    else
                        sind_d = 1d0
                    end if
                    costp = cost * dcos(d) - red(pix) * sind_d * sint
                    thtp = dacos(costp)
                    phip = modulo(phi + dasin(imd(pix) / dsqrt(1. - costp * costp) * sind_d), PI2) ! ok except for absurdly large d
                    gamma = datan2(imd(pix), red(pix)) - datan2(imd(pix), d * sind_d * d * cott + red(pix) * dcos(d))
                    ptg(1, pix) = thtp
                    ptg(2, pix) = phip
                    ptg(3, pix) = gamma
                    pix = pix + 1
                    phi = phi  + dphi
                end do
            else if (version == 1) then
                e_t = 2d0 * dsin(thts(ir) * 0.5d0) ** 2
                do ip = 1, nphis(ir)
                    d = red(pix) * red(pix) + imd(pix) * imd(pix)
                    if (d > 0d0) then
                        d = dsqrt(d)
                        sind_d = dsin(d) / d
                    else
                        sind_d = 1d0
                    end if
                    e_d = 2d0 * dsin(d * 0.5d0) ** 2
                    e_tp = e_t + e_d - e_t * e_d + red(pix) * sind_d * sint
                    sintp = dsqrt(dmax1(0d0, e_tp * (2d0 - e_tp)))
                    thtp = dasin(sintp)
                    phip = modulo(phi + datan2(imd(pix) * sind_d, (1d0 - e_d) * sint + red(pix) * sind_d * (1d0 - e_t)), PI2)
                    gamma = datan2(imd(pix), red(pix)) - datan2(imd(pix), d * sind_d * d * cott + red(pix) * dcos(d))
                    ptg(1, pix) = thtp
                    ptg(2, pix) = phip
                    ptg(3, pix) = gamma
                    pix = pix + 1
                    phi = phi  + dphi
                end do
            else if (version == -1) then
                e_t = 2d0 * dcos(thts(ir) * 0.5d0) ** 2
                do ip = 1, nphis(ir)
                    d = red(pix) * red(pix) + imd(pix) * imd(pix)
                    if (d > 0d0) then
                        d = dsqrt(d)
                        sind_d = dsin(d) / d
                    else
                        sind_d = 1d0
                    end if
                    e_d = 2d0 * dsin(d * 0.5d0) ** 2
                    e_tp = e_t + e_d - e_t * e_d - red(pix) * sind_d * sint
                    sintp = dsqrt(dmax1(0d0, e_tp * (2d0 - e_tp)))
                    thtp = DPI - dasin(sintp)
                    phip = modulo(phi + datan2(imd(pix) * sind_d, (1d0 - e_d) * sint + red(pix) * sind_d * (e_t - 1d0)), PI2)
                    gamma = datan2(imd(pix), red(pix)) - datan2(imd(pix), d * sind_d * d * cott + red(pix) * dcos(d))
                    ptg(1, pix) = thtp
                    ptg(2, pix) = phip
                    ptg(3, pix) = gamma
                    pix = pix + 1
                    phi = phi  + dphi
                end do
            end if
        end do
        !$OMP END PARALLEL DO
    end subroutine fpointing
    subroutine pointing(npix, nring, red, imd, thts, phi0s, nphis, ptg, ofs, nthreads)
        use OMP_LIB
        implicit none
        integer, intent(in) :: npix, nring, nthreads
        integer, intent(in) :: nphis(nring), ofs(nring)
        double precision, intent(in) :: red(npix), imd(npix), thts(nring), phi0s(nring)
        double precision, intent(out) :: ptg(3, npix)
        double precision thtp, phip, gamma
        double precision phi, dphi, sint, cost, cott, d, sind_d, e_t, e_d, e_tp, costp, sintp
        double precision :: PI2 = DPI * 2d0
        integer ir, ip, pix, version

        call OMP_SET_NUM_THREADS(nthreads)
        !$OMP PARALLEL DO DEFAULT(NONE)&
        !$OMP SHARED(PI2, thts, red, imd, ptg, nphis, phi0s, npix, nring, ofs)&
        !$OMP PRIVATE(ip, thtp, phip,gamma, e_t, sint, cost, cott, e_d, e_tp, costp, sintp, pix, phi, dphi, version, d, sind_d)
        do ir = 1, nring
            sint = dsin(thts(ir))
            cost = dcos(thts(ir))
            cott = cost / sint
            version = nint(1d0 - 2d0 * thts(ir) / DPI)
            pix = ofs(ir) + 1
            phi = phi0s(ir)
            dphi =  PI2  / nphis(ir)
            if (version == 0) then
                do ip = 1, nphis(ir)
                    d = red(pix) * red(pix) + imd(pix) * imd(pix)
                    !sind_d = 1d0 - d / 6d0 * (1d0 - d / 20d0 * (1d0 - d / 42d0))
                    !d = dsqrt(d)
                    if (d > 0d0) then
                        d = dsqrt(d)
                        sind_d = dsin(d) / d
                    else
                        sind_d = 1d0
                    end if
                    costp = cost * dcos(d) - red(pix) * sind_d * sint
                    thtp = dacos(costp)
                    phip = modulo(phi + dasin(imd(pix) / dsqrt(1. - costp * costp) * sind_d), PI2) ! ok except for absurdly large d
                    gamma = datan2(imd(pix), red(pix)) - datan2(imd(pix), d * sind_d * d * cott + red(pix) * dcos(d))
                    ptg(1, pix) = thtp
                    ptg(2, pix) = phip
                    ptg(3, pix) = gamma
                    pix = pix + 1
                    phi = phi  + dphi
                end do
            else if (version == 1) then
                e_t = 2d0 * dsin(thts(ir) * 0.5d0) ** 2
                do ip = 1, nphis(ir)
                    d = red(pix) * red(pix) + imd(pix) * imd(pix)
                    if (d > 0d0) then
                        d = dsqrt(d)
                        sind_d = dsin(d) / d
                    else
                        sind_d = 1d0
                    end if
                    e_d = 2d0 * dsin(d * 0.5d0) ** 2
                    e_tp = e_t + e_d - e_t * e_d + red(pix) * sind_d * sint
                    sintp = dsqrt(dmax1(0d0, e_tp * (2d0 - e_tp)))
                    thtp = dasin(sintp)
                    phip = modulo(phi + datan2(imd(pix) * sind_d, (1d0 - e_d) * sint + red(pix) * sind_d * (1d0 - e_t)), PI2)
                    gamma = datan2(imd(pix), red(pix)) - datan2(imd(pix), d * sind_d * d * cott + red(pix) * dcos(d))
                    ptg(1, pix) = thtp
                    ptg(2, pix) = phip
                    ptg(3, pix) = gamma
                    pix = pix + 1
                    phi = phi  + dphi
                end do
            else if (version == -1) then
                e_t = 2d0 * dcos(thts(ir) * 0.5d0) ** 2
                do ip = 1, nphis(ir)
                    d = red(pix) * red(pix) + imd(pix) * imd(pix)
                    if (d > 0d0) then
                        d = dsqrt(d)
                        sind_d = dsin(d) / d
                    else
                        sind_d = 1d0
                    end if
                    e_d = 2d0 * dsin(d * 0.5d0) ** 2
                    e_tp = e_t + e_d - e_t * e_d - red(pix) * sind_d * sint
                    sintp = dsqrt(dmax1(0d0, e_tp * (2d0 - e_tp)))
                    thtp = DPI - dasin(sintp)
                    phip = modulo(phi + datan2(imd(pix) * sind_d, (1d0 - e_d) * sint + red(pix) * sind_d * (e_t - 1d0)), PI2)
                    gamma = datan2(imd(pix), red(pix)) - datan2(imd(pix), d * sind_d * d * cott + red(pix) * dcos(d))
                    ptg(1, pix) = thtp
                    ptg(2, pix) = phip
                    ptg(3, pix) = gamma
                    pix = pix + 1
                    phi = phi  + dphi
                end do
            end if
        end do
        !$OMP END PARALLEL DO
    end subroutine pointing
    subroutine get_drange(npix, nring, red, imd, nphis, ofs, nthreads, drange)
        !gets min and max of deflection angle ring per ring
        use OMP_LIB
        implicit none
        integer, intent(in) :: npix, nring, nthreads
        integer, intent(in) :: nphis(nring), ofs(nring)
        double precision, intent(in) :: red(npix), imd(npix)
        double precision, intent(out) :: drange(nring, 2)
        integer ir, ip, pix
        double precision a, amin, amax
        call OMP_SET_NUM_THREADS(nthreads)
        !$OMP PARALLEL DO
        do ir = 1, nring
            amin = 1d30
            amax = 0d0
            do ip = 1, nphis(ir)
                pix = ofs(ir) + ip
                a = red(pix) * red(pix) + imd(pix) * imd(pix)
                amin = MIN(a, amin)
                amax = MAX(a, amax)
            end do
            drange(ir, 1) = dsqrt(amin)
            drange(ir, 2) = dsqrt(amax)
        end do
        !$OMP END PARALLEL DO
    end subroutine get_drange

    subroutine apply_inplace(npix, values, gamma, spin, nthreads)
        use OMP_LIB
        implicit none
        double precision, intent(in) :: gamma(npix)
        integer, intent(in) :: npix, spin, nthreads
        double complex, intent(inout) :: values(npix)
        integer i

        call OMP_SET_NUM_THREADS(nthreads)
        !$OMP PARALLEL DO
        ! spin must be 'shared' here for things to work, no idea why
        do i = 1, npix
            values(i) = values(i) * cdexp(dcmplx(0, spin * gamma(i)))
        end do
        !$OMP END PARALLEL DO
    end subroutine apply_inplace
    subroutine apply_inplacef(npix, values, gamma, spin, nthreads)
        use OMP_LIB
        implicit none
        real, intent(in) :: gamma(npix)
        integer, intent(in) :: npix, spin, nthreads
        complex, intent(inout) :: values(npix)
        integer i

        call OMP_SET_NUM_THREADS(nthreads)
        !$OMP PARALLEL DO
        do i = 1, npix
            values(i) = values(i) * cexp(complex(0, spin * gamma(i)))
        end do
        !$OMP END PARALLEL DO
    end subroutine apply_inplacef
    subroutine my_incr(var, incr)
      implicit none
      integer, intent(inout) :: var
      integer, intent(in)    :: incr
      var = var + incr
    end subroutine my_incr
end module remapping
