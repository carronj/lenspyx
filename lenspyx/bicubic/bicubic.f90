double precision function cubicfilter(x, c0, c1, c2, c3)
    ! filter 4 values using cubic splines
    double precision x, c0, c1, c2, c3
    double precision w0, w1, w2, w3
    w0 = x*(x*(-x + 3d0) - 3d0) + 1d0
    w1 = x*x*(3d0*x - 6d0) + 4d0;
    w2 =  x*(x*(-3d0*x + 3d0) + 3d0) + 1d0
    w3 =  x*x*x
    cubicfilter = (c0 * w0 + c1 * w1 + c2 * w2 + c3 * w3) * 0.16666666666666666d0
end

double precision function tex2d(ftl_map, i, j, nx, ny)
    ! nx : second dimension of array
    ! ny : first dimension of array
    double precision, intent(in) :: ftl_map(0:ny-1, 0:nx-1)
    integer, intent(in) :: nx, ny
    integer i, j

    if (i < 0)  i = mod(ny + i, ny)
    if (j < 0)  j = mod(nx + j, nx)
    if (i >= ny) i = mod(i, ny)
    if (j >= nx) j = mod(j, nx)
    tex2d = ftl_map(i, j)
end

double precision function eval(ftl_map, fx, fy, nx, ny)
    ! nx : second dimension of array
    ! ny : first dimension of array
    implicit none
    double precision, intent(in) :: ftl_map(0:ny-1, 0:nx-1), fx, fy
    integer, intent(in) :: nx, ny
    double precision, external :: cubicfilter, tex2d
    double precision gx, gy
    integer px, py

    px = floor(fx)
    py = floor(fy)
    gx = fx - px
    gy = fy - py
    eval = cubicfilter(gy, &
              cubicfilter(gx, tex2d(ftl_map, px-1, py-1,nx, ny), tex2d(ftl_map, px, py-1,nx, ny), &
                        tex2d(ftl_map, px+1, py-1,nx, ny), tex2d(ftl_map, px+2,py-1,nx, ny)), &
              cubicfilter(gx, tex2D(ftl_map, px-1, py, nx, ny),   tex2d(ftl_map, px, py,nx, ny),   &
                        tex2d(ftl_map, px+1, py, nx, ny),   tex2d(ftl_map, px+2, py,nx, ny)),&
              cubicfilter(gx, tex2D(ftl_map, px-1, py+1,nx, ny), tex2d(ftl_map, px, py+1,nx, ny), &
                        tex2d(ftl_map, px+1, py+1, nx, ny), tex2d(ftl_map, px+2, py+1,nx, ny)), &
              cubicfilter(gx, tex2D(ftl_map, px-1, py+2,nx, ny), tex2d(ftl_map, px, py+2,nx, ny), &
                        tex2d(ftl_map, px+1, py+2,nx, ny), tex2d(ftl_map, px+2, py+2,nx, ny)) )
end

integer function mod_test(i, npix)
    integer i, npix
    mod_test = i
    if (mod_test < 0) mod_test = mod(npix + mod_test, npix)
    if (mod_test >= npix)  mod_test = mod(mod_test, npix)
end function mod_test

subroutine deflect(output, ftl_map, fx, fy, nx, ny, npts)
    ! input ftl_map should be bicubic prefiltered map
    ! fx, fy new coordinate in grid units.
    !FIXME: misleading notation. fx is deflection in first ftl_map array dimension, fy in second dimension
    !FIXME: but nx is second dimension and ny first dimension.
    !FIXME: should rename fx <-> fy here and in eval and in tex2d

    implicit none
    double precision, intent(in) :: ftl_map(0:ny-1,0:nx-1)
    double precision, intent(in) :: fx(0:npts-1), fy(0:npts-1)
    double precision, intent(out) :: output(0:npts-1)
    double precision, external :: eval
    integer, intent(in) :: nx, ny, npts
    integer i
    do i = 0, npts - 1
        output(i) = eval(ftl_map, fx(i), fy(i), nx, ny)
    end do
end subroutine deflect
