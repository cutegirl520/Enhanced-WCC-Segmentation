*> \brief \b DBLAT1
*
*  =========== DOCUMENTATION ===========
*
* Online html documentation available at 
*            http://www.netlib.org/lapack/explore-html/ 
*
*  Definition:
*  ===========
*
*       PROGRAM DBLAT1
* 
*
*> \par Purpose:
*  =============
*>
*> \verbatim
*>
*>    Test program for the DOUBLE PRECISION Level 1 BLAS.
*>
*>    Based upon the original BLAS test routine together with:
*>    F06EAF Example Program Text
*> \endverbatim
*
*  Authors:
*  ========
*
*> \author Univ. of Tennessee 
*> \author Univ. of California Berkeley 
*> \author Univ. of Colorado Denver 
*> \author NAG Ltd. 
*
*> \date April 2012
*
*> \ingroup double_blas_testing
*
*  =====================================================================
      PROGRAM DBLAT1
*
*  -- Reference BLAS test routine (version 3.4.1) --
*  -- Reference BLAS is a software package provided by Univ. of Tennessee,    --
*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
*     April 2012
*
*  =====================================================================
*
*     .. Parameters ..
      INTEGER          NOUT
      PARAMETER        (NOUT=6)
*     .. Scalars in Common ..
      INTEGER          ICASE, INCX, INCY, N
      LOGICAL          PASS
*     .. Local Scalars ..
      DOUBLE PRECISION SFAC
      INTEGER          IC
*     .. External Subroutines ..
      EXTERNAL         CHECK0, CHECK1, CHECK2, CHECK3, HEADER
*     .. Common blocks ..
      COMMON           /COMBLA/ICASE, N, INCX, INCY, PASS
*     .. Data statements ..
      DATA             SFAC/9.765625D-4/
*     .. Executable Statements ..
      WRITE (NOUT,99999)
      DO 20 IC = 1, 13
         ICASE = IC
         CALL HEADER
*
*        .. Initialize  PASS,  INCX,  and INCY for a new case. ..
*        .. the value 9999 for INCX or INCY will appear in the ..
*        .. detailed  output, if any, for cases  that do not involve ..
*        .. these parameters ..
*
         PASS = .TRUE.
         INCX = 9999
         INCY = 9999
         IF (ICASE.EQ.3 .OR. ICASE.EQ.11) THEN
            CALL CHECK0(SFAC)
         ELSE IF (ICASE.EQ.7 .OR. ICASE.EQ.8 .OR. ICASE.EQ.9 .OR.
     +            ICASE.EQ.10) THEN
            CALL CHECK1(SFAC)
         ELSE IF (ICASE.EQ.1 .OR. ICASE.EQ.2 .OR. ICASE.EQ.5 .OR.
     +            ICASE.EQ.6 .OR. ICASE.EQ.12 .OR. ICASE.EQ.13) THEN
            CALL CHECK2(SFAC)
         ELSE IF (ICASE.EQ.4) THEN
            CALL CHECK3(SFAC)
         END IF
*        -- Print
         IF (PASS) WRITE (NOUT,99998)
   20 CONTINUE
      STOP
*
99999 FORMAT (' Real BLAS Test Program Results',/1X)
99998 FORMAT ('                                    ----- PASS -----')
      END
      SUBROUTINE HEADER
*     .. Parameters ..
      INTEGER          NOUT
      PARAMETER        (NOUT=6)
*     .. Scalars in Common ..
      INTEGER          ICASE, INCX, INCY, N
      LOGICAL          PASS
*     .. Local Arrays ..
      CHARACTER*6      L(13)
*     .. Common blocks ..
      COMMON           /COMBLA/ICASE, N, INCX, INCY, PASS
*     .. Data statements ..
      DATA             L(1)/' DDOT '/
      DATA             L(2)/'DAXPY '/
      DATA             L(3)/'DROTG '/
      DATA             L(4)/' DROT '/
      DATA             L(5)/'DCOPY '/
      DATA             L(6)/'DSWAP '/
      DATA             L(7)/'DNRM2 '/
      DATA             L(8)/'DASUM '/
      DATA             L(9)/'DSCAL '/
      DATA             L(10)/'IDAMAX'/
      DATA             L(11)/'DROTMG'/
      DATA             L(12)/'DROTM '/
      DATA             L(13)/'DSDOT '/
*     .. Executable Statements ..
      WRITE (NOUT,99999) ICASE, L(ICASE)
      RETURN
*
99999 FORMAT (/' Test of subprogram number',I3,12X,A6)
      END
      SUBROUTINE CHECK0(SFAC)
*     .. Parameters ..
      INTEGER           NOUT
      PARAMETER         (NOUT=6)
*     .. Scalar Arguments ..
      DOUBLE PRECISION  SFAC
*     .. Scalars in Common ..
      INTEGER           ICASE, INCX, INCY, N
      LOGICAL           PASS
*     .. Local Scalars ..
      DOUBLE PRECISION  SA, SB, SC, SS, D12
      INTEGER           I, K
*     .. Local Arrays ..
      DOUBLE PRECISION  DA1(8), DATRUE(8), DB1(8), DBTRUE(8), DC1(8),
     $                  DS1(8), DAB(4,9), DTEMP(9), DTRUE(9,9)
*     .. External Subroutines ..
      EXTERNAL          DROTG, DROTMG, STEST1
*     .. Common blocks ..
      COMMON            /COMBLA/ICASE, N, INCX, INCY, PASS
*     .. Data statements ..
      DATA              DA1/0.3D0, 0.4D0, -0.3D0, -0.4D0, -0.3D0, 0.0D0,
     +                  0.0D0, 1.0D0/
      DATA              DB1/0.4D0, 0.3D0, 0.4D0, 0.3D0, -0.4D0, 0.0D0,
     +                  1.0D0, 0.0D0/
      DATA              DC1/0.6D0, 0.8D0, -0.6D0, 0.8D0, 0.6D0, 1.0D0,
     +                  0.0D0, 1.0D0/
      DATA              DS1/0.8D0, 0.6D0, 0.8D0, -0.6D0, 0.8D0, 0.0D0,
     +                  1.0D0, 0.0D0/
      DATA              DATRUE/0.5D0, 0.5D0, 0.5D0, -0.5D0, -0.5D0,
     +                  0.0D0, 1.0D0, 1.0D0/
      DATA              DBTRUE/0.0D0, 0.6D0, 0.0D0, -0.6D0, 0.0D0,
     +                  0.0D0, 1.0D0, 0.0D0/
*     INPUT FOR MODIFIED GIVENS
      DATA DAB/ .1D0,.3D0,1.2D0,.2D0,
     A          .7D0, .2D0, .6D0, 4.2D0,
     B          0.D0,0.D0,0.D0,0.D0,
     C          4.D0, -1.D0, 2.D0, 4.D0,
     D          6.D-10, 2.D-2, 1.D5, 10.D0,
     E          4.D10, 2.D-2, 1.D-5, 10.D0,
     F          2.D-10, 4.D-2, 1.D5, 10.D0,
     G          2.D10, 4.D-2, 1.D-5, 10.D0,
     H          4.D0, -2.D0, 8.D0, 4.D0    /
*    TRUE RESULTS FOR MODIFIED GIVENS
      DATA DTRUE/0.D0,0.D0, 1.3D0, .2D0, 0.D0,0.D0,0.D0, .5D0, 0.D0,
     A           0.D0,0.D0, 4.5D0, 4.2D0, 1.D0, .5D0, 0.D0,0.D0,0.D0,
     B           0.D0,0.D0,0.D0,0.D0, -2.D0, 0.D0,0.D0,0.D0,0.D0,
     C           0.D0,0.D0,0.D0, 4.D0, -1.D0, 0.D0,0.D0,0.D0,0.D0,
     D           0.D0, 15.D-3, 0.D0, 10.D0, -1.D0, 0.D0, -1.D-4,
     E           0.D0, 1.D0,
     F           0.D0,0.D0, 6144.D-5, 10.D0, -1.D0, 4096.D0, -1.D6,
     G           0.D0, 1.D0,
     H           0.D0,0.D0,15.D0,10.D0,-1.D0, 5.D-5, 0.D0,1.D0,0.D0,
     I           0.D0,0.D0, 15.D0, 10.D0, -1. D0, 5.D5, -4096.D0,
     J           1.D0, 4096.D-6,
     K           0.D0,0.D0, 7.D0, 4.D0, 0.D0,0.D0, -.5D0, -.25D0, 0.D0/
*                   4096 = 2 ** 12
      DATA D12  /4096.D0/
      DTRUE(1,1) = 12.D0 / 130.D0
      DTRUE(2,1) = 36.D0 / 130.D0
      DTRUE(7,1) = -1.D0 / 6.D0
      DTRUE(1,2) = 14.D0 / 75.D0
      DTRUE(2,2) = 49.D0 / 75.D0
      DTRUE(9,2) = 1.D0 / 7.D0
      DTRUE(1,5) = 45.D-11 * (D12 * D12)
      DTRUE(3,5) = 4.D5 / (3.D0 * D12)
      DTRUE(6,5) = 1.D0 / D12
      DTRUE(8,5) = 1.D4 / (3.D0 * D12)
      DTRUE(1,6) = 4.D10 / (1.5D0 * D12 * D12)
      DTRUE(2,6) = 2.D-2 / 1.5D0
      DTRUE(8,6) = 5.D-7 * D12
      DTRUE(1,7) = 4.D0 / 150.D0
      DTRUE(2,7) = (2.D-10 / 1.5D0) * (D12 * D12)
      DTRUE(7,7) = -DTRUE(6,5)
      DTRUE(9,7) = 1.D4 / D12
      DTRUE(1,8) = DTRUE(1,7)
      DTRUE(2,8) = 2.D10 / (1.5D0 * D12 * D12)
      DTRUE(1,9) = 32.D0 / 7.D0
      DTRUE(2,9) = -16.D0 / 7.D0
*     .. Executable Statements ..
*
*     Compute true values which cannot be prestored
*     in decimal notation
*
      DBTRUE(1) = 1.0D0/0.6D0
      DBTRUE(3) = -1.0D0/0.6D0
      DBTRUE(5) = 1.0D0/0.6D0
*
      DO 20 K = 1, 8
*        .. Set N=K for identification in output if any ..
         N = K
         IF (ICASE.EQ.3) THEN
*           .. DROTG ..
            IF (K.GT.8) GO TO 40
            SA = DA1(K)
            SB = DB1(K)
            CALL DROTG(SA,SB,SC,SS)
            CALL STEST1(SA,DATRUE(K),DATRUE(K),SFAC)
            CALL STEST1(SB,DBTRUE(K),DBTRUE(K),SFAC)
            CALL STEST1(SC,DC1(K),DC1(K),SFAC)
            CALL STEST1(SS,DS1(K),DS1(K),SFAC)
         ELSEIF (ICASE.EQ.11) THEN
*           .. DROTMG ..
            DO I=1,4
               DTEMP(I)= DAB(I,K)
               DTEMP(I+4) = 0.0
            END DO
            DTEMP(9) = 0.0
            CALL DROTMG(DTEMP(1),DTEMP(2),DTEMP(3),DTEMP(4),DTEMP(5))
            CALL STEST(9,DTEMP,DTRUE(1,K),DTRUE(1,K),SFAC)
         ELSE
            WRITE (NOUT,*) ' Shouldn''t be here in CHECK0'
            STOP
         END IF
   20 CONTINUE
   40 RETURN
      END
      SUBROUTINE CHECK1(SFAC)
*     .. Parameters ..
      INTEGER           NOUT
      PARAMETER         (NOUT=6)
*     .. Scalar Arguments ..
      DOUBLE PRECISION  SFAC
*     .. Scalars in Common ..
      INTEGER           ICASE, INCX, INCY, N
      LOGICAL           PASS
*     .. Local Scalars ..
      INTEGER           I, LEN, NP1
*     .. Local Arrays ..
      DOUBLE PRECISION  DTRUE1(5), DTRUE3(5), DTRUE5(8,5,2), DV(8,5,2),
     +                  SA(10), STEMP(1), STRUE(8), SX(8)
      INTEGER           ITRUE2(5)
*     .. External Functions ..
      DOUBLE PRECISION  DASUM, DNRM2
      INTEGER           IDAMAX
      EXTERNAL          DASUM, DNRM2, IDAMAX
*     .. External Subroutines ..
      EXTERNAL          ITEST1, DSCAL, STEST, STEST1
*     .. Intrinsic Functions ..
      INTRINSIC         MAX
*     .. Common blocks ..
      COMMON            /COMBLA/ICASE, N, INCX, INCY, PASS
*     .. Data statements ..
      DATA              SA/0.3D0, -1.0D0, 0.0D0, 1.0D0, 0.3D0, 0.3D0,
     +                  0.3D0, 0.3D0, 0.3D0, 0.3D0/
      DATA              DV/0.1D0, 2.0D0, 2.0D0, 2.0D0, 2.0D0, 2.0D0,
     +                  2.0D0, 2.0D0, 0.3D0, 3.0D0, 3.0D0, 3.0D0, 3.0D0,
     +                  3.0D0, 3.0D0, 3.0D0, 0.3D0, -0.4D0, 4.0D0,
     +                  4.0D0, 4.0D0, 4.0D0, 4.0D0, 4.0D0, 0.2D0,
     +                  -0.6D0, 0.3D0, 5.0D0, 5.0D0, 5.0D0, 5.0D0,
     +                  5.0D0, 0.1D0, -0.3D0, 0.5D0, -0.1D0, 6.0D0,
     +                  6.0D0, 6.0D0, 6.0D0, 0.1D0, 8.0D0, 8.0D0, 8.0D0,
     +                  8.0D0, 8.0D0, 8.0D0, 8.0D0, 0.3D0, 9.0D0, 9.0D0,
     +                  9.0D0, 9.0D0, 9.0D0, 9.0D0, 9.0D0, 0.3D0, 2.0D0,
     +                  -0.4D0, 2.0D0, 2.0D0, 2.0D0, 2.0D0, 2.0D0,
     +                  0.2D0, 3.0D0, -0.6D0, 5.0D0, 0.3D0, 2.0D0,
     +                  2.0D0, 2.0D0, 0.1D0, 4.0D0, -0.3D0, 6.0D0,
     +                  -0.5D0, 7.0D0, -0.1D0, 3.0D0/
      DATA              DTRUE1/0.0D0, 0.3D0, 0.5D0, 0.7D0, 0.6D0/
      DATA              DTRUE3/0.0D0, 0.3D0, 0.7D0, 1.1D0, 1.0D0/
      DATA              DTRUE5/0.10D0, 2.0D0, 2.0D0, 2.0D0, 2.0D0,
     +                  2.0D0, 2.0D0, 2.0D0, -0.3D0, 3.0D0, 3.0D0,
     +                  3.0D0, 3.0D0, 3.0D0, 3.0D0, 3.0D0, 0.0D0, 0.0D0,
     +                  4.0D0, 4.0D0, 4.0D0, 4.0D0, 4.0D0, 4.0D0,
     +                  0.20D0, -0.60D0, 0.30D0, 5.0D0, 5.0D0, 5.0D0,
     +                  5.0D0, 5.0D0, 0.03D0, -0.09D0, 0.15D0, -0.03D0,
     +                  6.0D0, 6.0D0, 6.0D0, 6.0D0, 0.10D0, 8.0D0,
     +                  8.0D0, 8.0D0, 8.0D0, 8.0D0, 8.0D0, 8.0D0,
     +                  0.09D0, 9.0D0, 9.0D0, 9.0D0, 9.0D0, 9.0D0,
     +                  9.0D0, 9.0D0, 0.09D0, 2.0D0, -0.12D0, 2.0D0,
     +                  2.0D0, 2.0D0, 2.0D0, 2.0D0, 0.06D0, 3.0D0,
     +                  -0.18D0, 5.0D0, 0.09D0, 2.0D0, 2.0D0, 2.0D0,
     +                  0.03D0, 4.0D0, -0.09D0, 6.0D0, -0.15D0, 7.0D0,
     +                  -0.03D0, 3.0D0/
      DATA              ITRUE2/0, 1, 2, 2, 3/
*     .. Executable Statements ..
      DO 80 INCX = 1, 2
         DO 60 NP1 = 1, 5
            N = NP1 - 1
            LEN = 2*MAX(N,1)
*           .. Set vector arguments ..
            DO 20 I = 1, LEN
               SX(I) = DV(I,NP1,INCX)
   20       CONTINUE
*
            IF (ICASE.EQ.7) THEN
*              .. DNRM2 ..
               STEMP(1) = DTRUE1(NP1)
               CALL STEST1(DNRM2(N,SX,INCX),STEMP(1),STEMP,SFAC)
            ELSE IF (ICASE.EQ.8) THEN
*              .. DASUM ..
               STEMP(1) = DTRUE3(NP1)
               CALL STEST1(DASUM(N,SX,INCX),STEMP(1),STEMP,SFAC)
            ELSE IF (ICASE.EQ.9) THEN
*              .. DSCAL ..
               CALL DSCAL(N,SA((INCX-1)*5+NP1),SX,INCX)
               DO 40 I = 1, LEN
                  STRUE(I) = DTRUE5(I,NP1,INCX)
   40          CONTINUE
               CALL STEST(LEN,SX,STRUE,STRUE,SFAC)
            ELSE IF (ICASE.EQ.10) THEN
*              .. IDAMAX ..
               CALL ITEST1(IDAMAX(N,SX,INCX),ITRUE2(NP1))
            ELSE
               WRITE (NOUT,*) ' Shouldn''t be here in CHECK1'
               STOP
            END IF
   60    CONTINUE
   80 CONTINUE
      RETURN
      END
      SUBROUTINE CHECK2(SFAC)
*     .. Parameters ..
      INTEGER           NOUT
      PARAMETER         (NOUT=6)
*     .. Scalar Arguments ..
      DOUBLE PRECISION  SFAC
*     .. Scalars in Common ..
      INTEGER           ICASE, INCX, INCY, N
      LOGICAL           PASS
*     .. Local Scalars ..
      DOUBLE PRECISION  SA
      INTEGER           I, J, KI, KN, KNI, KPAR, KSIZE, LENX, LENY,
     $                  MX, MY 
*     .. Local Arrays ..
      DOUBLE PRECISION  DT10X(7,4,4), DT10Y(7,4,4), DT7(4,4),
     $                  DT8(7,4,4), DX1(7),
     $                  DY1(7), SSIZE1(4), SSIZE2(14,2), SSIZE(7),
     $                  STX(7), STY(7), SX(7), SY(7),
     $                  DPAR(5,4), DT19X(7,4,16),DT19XA(7,4,4),
     $                  DT19XB(7,4,4), DT19XC(7,4,4),DT19XD(7,4,4),
     $                  DT19Y(7,4,16), DT19YA(7,4,4),DT19YB(7,4,4),
     $                  DT19YC(7,4,4), DT19YD(7,4,4), DTEMP(5)
      INTEGER           INCXS(4), INCYS(4), LENS(4,2), NS(4)
*     .. External Functions ..
      DOUBLE PRECISION  DDOT, DSDOT
      EXTERNAL          DDOT, DSDOT
*     .. External Subroutines ..
      EXTERNAL          DAXPY, DCOPY, DROTM, DSWAP, STEST, STEST1
*     .. Intrinsic Functions ..
      INTRINSIC         ABS, MIN
*     .. Common blocks ..
      COMMON            /COMBLA/ICASE, N, INCX, INCY, PASS
*     .. Data statements ..
      EQUIVALENCE (DT19X(1,1,1),DT19XA(1,1,1)),(DT19X(1,1,5),
     A   DT19XB(1,1,1)),(DT19X(1,1,9),DT19XC(1,1,1)),
     B   (DT19X(1,1,13),DT19XD(1,1,1))
      EQUIVALENCE (DT19Y(1,1,1),DT19YA(1,1,1)),(DT19Y(1,1,5),
     A   DT19YB(1,1,1)),(DT19Y(1,1,9),DT19YC(1,1,1)),
     B   (DT19Y(1,1,13),DT19YD(1,1,1))

      DATA              SA/0.3D0/
      DATA              INCXS/1, 2, -2, -1/
      DATA              INCYS/1, -2, 1, -2/
      DATA              LENS/1, 1, 2, 4, 1, 1, 3, 7/
      DATA              NS/0, 1, 2, 4/
      DATA              DX1/0.6D0, 0.1D0, -0.5D0, 0.8D0, 0.9D0, -0.3D0,
     +                  -0.4D0/
      DATA              DY1/0.5D0, -0.9D0, 0.3D0, 0.7D0, -0.6D0, 0.2D0,
     +                  0.8D0/
      DATA              DT7/0.0D0, 0.30D0, 0.21D0, 0.62D0, 0.0D0,
     +                  0.30D0, -0.07D0, 0.85D0, 0.0D0, 0.30D0, -0.79D0,
     +                  -0.74D0, 0.0D0, 0.30D0, 0.33D0, 1.27D0/
      DATA              DT8/0.5D0, 0.0D0, 0.0D0, 0.0D0, 0.0D0, 0.0D0,
     +                  0.0D0, 0.68D0, 0.0D0, 0.0D0, 0.0D0, 0.0D0,
     +                  0.0D0, 0.0D0, 0.68D0, -0.87D0, 0.0D0, 0.0D0,
     +                  0.0D0, 0.0D0, 0.0D0, 0.68D0, -0.87D0, 0.15D0,
     +                  0.94D0, 0.0D0, 0.0D0, 0.0D0, 0.5D0, 0.0D0,
     +                  0.0D0, 0.0D0, 0.0D0, 0.0D0, 0.0D0, 0.68D0,
     +                  0.0D0, 0.0D0, 0.0D0, 0.0D0, 0.0D0, 0.0D0,
     +                  0.35D0, -0.9D0, 0.48D0, 0.0D0, 0.0D0, 0.0D0,
     +                  0.0D0, 0.38D0, -0.9D0, 0.57D0, 0.7D0, -0.75D0,
     +                  0.2D0, 0.98D0, 0.5D0, 0.0D0, 0.0D0, 0.0D0,
     +                  0.0D0, 0.0D0, 0.0D0, 0.68D0, 0.0D0, 0.0D0,
     +                  0.0D0, 0.0D0, 0.0D0, 0.0D0, 0.35D0, -0.72D0,
     +                  0.0D0, 0.0D0, 0.0D0, 0.0D0, 0.0D0, 0.38D0,
     +                  -0.63D0, 0.15D0, 0.88D0, 0.0D0, 0.0D0, 0.0D0,
     +                  0.5D0, 0.0D0, 0.0D0, 0.0D0, 0.0D0, 0.0D0, 0.0D0,
     +                  0.68D0, 0.0D0, 0.0D0, 0.0D0, 0.0D0, 0.0D0,
     +                  0.0D0, 0.68D0, -0.9D0, 0.33D0, 0.0D0, 0.0D0,
     +                  0.0D0, 0.0D0, 0.68D0, -0.9D0, 0.33D0, 0.7D0,
     +                  -0.75D0, 0.2D0, 1.04D0/
      DATA              DT10X/0.6D0, 0.0D0, 0.0D0, 0.0D0, 0.0D0, 0.0D0,
     +                  0.0D0, 0.5D0, 0.0D0, 0.0D0, 0.0D0, 0.0D0, 0.0D0,
     +                  0.0D0, 0.5D0, -0.9D0, 0.0D0, 0.0D0, 0.0D0,
     +                  0.0D0, 0.0D0, 0.5D0, -0.9D0, 0.3D0, 0.7D0,
     +                  0.0D0, 0.0D0, 0.0D0, 0.6D0, 0.0D0, 0.0D0, 0.0D0,
     +                  0.0D0, 0.0D0, 0.0D0, 0.5D0, 0.0D0, 0.0D0, 0.0D0,
     +                  0.0D0, 0.0D0, 0.0D0, 0.3D0, 0.1D0, 0.5D0, 0.0D0,
     +                  0.0D0, 0.0D0, 0.0D0, 0.8D0, 0.1D0, -0.6D0,
     +                  0.8D0, 0.3D0, -0.3D0, 0.5D0, 0.6D0, 0.0D0,
     +                  0.0D0, 0.0D0, 0.0D0, 0.0D0, 0.0D0, 0.5D0, 0.0D0,
     +                  0.0D0, 0.0D0, 0.0D0, 0.0D0, 0.0D0, -0.9D0,
     +                  0.1D0, 0.5D0, 0.0D0, 0.0D0, 0.0D0, 0.0D0, 0.7D0,
     +                  0.1D0, 0.3D0, 0.8D0, -0.9D0, -0.3D0, 0.5D0,
     +                  0.6D0, 0.0D0, 0.0D0, 0.0D0, 0.0D0, 0.0D0, 0.0D0,
     +                  0.5D0, 0.0D0, 0.0D0, 0.0D0, 0.0D0, 0.0D0, 0.0D0,
     +                  0.5D0, 0.3D0, 0.0D0, 0.0D0, 0.0D0, 0.0D0, 0.0D0,
     +                  0.5D0, 0.3D0, -0.6D0, 0.8D0, 0.0D0, 0.0D0,
     +                  0.0D0/
      DATA              DT10Y/0.5D0, 0.0D0, 0.0D0, 0.0D0, 0.0D0, 0.0D0,
     +                  0.0D0, 0.6D0, 0.0D0, 0.0D0, 0.0D0, 0.0D0, 0.0D0,
     +                  0.0D0, 0.6D0, 0.1D0, 0.0D0, 0.0D0, 0.0D0, 0.0D0,
     +                  0.0D0, 0.6D0, 0.1D0, -0.5D0, 0.8D0, 0.0D0,
     +                  0.0D0, 0.0D0, 0.5D0, 0.0D0, 0.0D0, 0.0D0, 0.0D0,
     +                  0.0D0, 0.0D0, 0.6D0, 0.0D0, 0.0D0, 0.0D0, 0.0D0,
     +                  0.0D0, 0.0D0, -0.5D0, -0.9D0, 0.6D0, 0.0D0,
     +                  0.0D0, 0.0D0, 0.0D0, -0.4D0, -0.9D0, 0.9D0,
     +                  0.7D0, -0.5D0, 0.2D0, 0.6D0, 0.5D0, 0.0D0,
     +                  0.0D0, 0.0D0, 0.0D0, 0.0D0, 0.0D0, 0.6D0, 0.0D0,
     +                  0.0D0, 0.0D0, 0.0D0, 0.0D0, 0.0D0, -0.5D0,
     +                  0.6D0, 0.0D0, 0.0D0, 0.0D0, 0.0D0, 0.0D0,
     +                  -0.4D0, 0.9D0, -0.5D0, 0.6D0, 0.0D0, 0.0D0,
     +                  0.0D0, 0.5D0, 0.0D0, 0.0D0, 0.0D0, 0.0D0, 0.0D0,
     +                  0.0D0, 0.6D0, 0.0D0, 0.0D0, 0.0D0, 0.0D0, 0.0D0,
     +                  0.0D0, 0.6D0, -0.9D0, 0.1D0, 0.0D0, 0.0D0,
     +                  0.0D0, 0.0D0, 0.6D0, -0.9D0, 0.1D0, 0.7D0,
     +                  -0.5D0, 0.2D0, 0.8D0/
      DATA              SSIZE1/0.0D0, 0.3D0, 1.6D0, 3.2D0/
      DATA              SSIZE2/0.0D0, 0.0D0, 0.0D0, 0.0D0, 0.0D0, 0.0D0,
     +                  0.0D0, 0.0D0, 0.0D0, 0.0D0, 0.0D0, 0.0D0, 0.0D0,
     +                  0.0D0, 1.17D0, 1.17D0, 1.17D0, 1.17D0, 1.17D0,
     +                  1.17D0, 1.17D0, 1.17D0, 1.17D0, 1.17D0, 1.17D0,
     +                  1.17D0, 1.17D0, 1.17D0/
*
*                         FOR DROTM
*
      DATA DPAR/-2.D0,  0.D0,0.D0,0.D0,0.D0,
     A          -1.D0,  2.D0, -3.D0, -4.D0,  5.D0,
     B           0.D0,  0.D0,  2.D0, -3.D0,  0.D0,
     C           1.D0,  5.D0,  2.D0,  0.D0, -4.D0/
*                        TRUE X RESULTS F0R ROTATIONS DROTM
      DATA DT19XA/.6D0,                  0.D0,0.D0,0.D0,0.D0,0.D0,0.D0,
     A            .6D0,                  0.D0,0.D0,0.D0,0.D0,0.D0,0.D0,
     B            .6D0,                  0.D0,0.D0,0.D0,0.D0,0.D0,0.D0,
     C            .6D0,                  0.D0,0.D0,0.D0,0.D0,0.D0,0.D0,
     D            .6D0,                  0.D0,0.D0,0.D0,0.D0,0.D0,0.D0,
     E           -.8D0,                  0.D0,0.D0,0.D0,0.D0,0.D0,0.D0,
     F           -.9D0,                  0.D0,0.D0,0.D0,0.D0,0.D0,0.D0,
     G           3.5D0,                  0.D0,0.D0,0.D0,0.D0,0.D0,0.D0,
     H            .6D0,   .1D0,             0.D0,0.D0,0.D0,0.D0,0.D0,
     I           -.8D0,  3.8D0,             0.D0,0.D0,0.D0,0.D0,0.D0,
     J           -.9D0,  2.8D0,             0.D0,0.D0,0.D0,0.D0,0.D0,
     K           3.5D0,  -.4D0,             0.D0,0.D0,0.D0,0.D0,0.D0,
     L            .6D0,   .1D0,  -.5D0,   .8D0,          0.D0,0.D0,0.D0,
     M           -.8D0,  3.8D0, -2.2D0, -1.2D0,          0.D0,0.D0,0.D0,
     N           -.9D0,  2.8D0, -1.4D0, -1.3D0,          0.D0,0.D0,0.D0,
     O           3.5D0,  -.4D0, -2.2D0,  4.7D0,          0.D0,0.D0,0.D0/
*
      DATA DT19XB/.6D0,                  0.D0,0.D0,0.D0,0.D0,0.D0,0.D0,
     A            .6D0,                  0.D0,0.D0,0.D0,0.D0,0.D0,0.D0,
     B            .6D0,                  0.D0,0.D0,0.D0,0.D0,0.D0,0.D0,
     C            .6D0,                  0.D0,0.D0,0.D0,0.D0,0.D0,0.D0,
     D            .6D0,                  0.D0,0.D0,0.D0,0.D0,0.D0,0.D0,
     E           -.8D0,                  0.D0,0.D0,0.D0,0.D0,0.D0,0.D0,
     F           -.9D0,                  0.D0,0.D0,0.D0,0.D0,0.D0,0.D0,
     G           3.5D0,                  0.D0,0.D0,0.D0,0.D0,0.D0,0.D0,
     H            .6D0,   .1D0,  -.5D0,             0.D0,0.D0,0.D0,0.D0,
     I           0.D0,    .1D0, -3.0D0,             0.D0,0.D0,0.D0,0.D0,
     J           -.3D0,   .1D0, -2.0D0,             0.D0,0.D0,0.D0,0.D0,
     K           3.3D0,   .1D0, -2.0D0,             0.D0,0.D0,0.D0,0.D0,
     L            .6D0,   .1D0,  -.5D0,   .8D0,   .9D0,  -.3D0,  -.4D0,
     M          -2.0D0,   .1D0,  1.4D0,   .8D0,   .6D0,  -.3D0, -2.8D0,
     N          -1.8D0,   .1D0,  1.3D0,   .8D0,  0.D0,   -.3D0, -1.9D0,
     O           3.8D0,   .1D0, -3.1D0,   .8D0,  4.8D0,  -.3D0, -1.5D0 /
*
      DATA DT19XC/.6D0,                  0.D0,0.D0,0.D0,0.D0,0.D0,0.D0,
     A            .6D0,                  0.D0,0.D0,0.D0,0.D0,0.D0,0.D0,
     B            .6D0,                  0.D0,0.D0,0.D0,0.D0,0.D0,0.D0,
     C            .6D0,                  0.D0,0.D0,0.D0,0.D0,0.D0,0.D0,
     D            .6D0,                  0.D0,0.D0,0.D0,0.D0,0.D0,0.D0,
     E           -.8D0,                  0.D0,0.D0,0.D0,0.D0,0.D0,0.D0,
     F           -.9D0,                  0.D0,0.D0,0.D0,0.D0,0.D0,0.D0,
     G           3.5D0,                  0.D0,0.D0,0.D0,0.D0,0.D0,0.D0,
     H            .6D0,   .1D0,  -.5D0,             0.D0,0.D0,0.D0,0.D0,
     I           4.8D0,   .1D0, -3.0D0,             0.D0,0.D0,0.D0,0.D0,
     J           3.3D0,   .1D0, -2.0D0,             0.D0,0.D0,0.D0,0.D0,
     K           2.1D0,   .1D0, -2.0D0,             0.D0,0.D0,0.D0,0.D0,
     L            .6D0,   .1D0,  -.5D0,   .8D0,   .9D0,  -.3D0,  -.4D0,
     M          -1.6D0,   .1D0, -2.2D0,   .8D0,  5.4D0,  -.3D0, -2.8D0,
     N          -1.5D0,   .1D0, -1.4D0,   .8D0,  3.6D0,  -.3D0, -1.9D0,
     O           3.7D0,   .1D0, -2.2D0,   .8D0,  3.6D0,  -.3D0, -1.5D0 /
*
      DATA DT19XD/.6D0,                  0.D0,0.D0,0.D0,0.D0,0.D0,0.D0,
     A            .6D0,                  0.D0,0.D0,0.D0,0.D0,0.D0,0.D0,
     B            .6D0,                  0.D0,0.D0,0.D0,0.D0,0.D0,0.D0,
     C            .6D0,                  0.D0,0.D0,0.D0,0.D0,0.D0,0.D0,
     D            .6D0,                  0.D0,0.D0,0.D0,0.D0,0.D0,0.D0,
     E           -.8D0,                  0.D0,0.D0,0.D0,0.D0,0.D0,0.D0,
     F           -.9D0,                  0.D0,0.D0,0.D0,0.D0,0.D0,0.D0,
     G           3.5D0,                  0.D0,0.D0,0.D0,0.D0,0.D0,0.D0,
     H            .6D0,   .1D0,             0.D0,0.D0,0.D0,0.D0,0.D0,
     I           -.8D0, -1.0D0,             0.D0,0.D0,0.D0,0.D0,0.D0,
     J           -.9D0,  -.8D0,             0.D0,0.D0,0.D0,0.D0,0.D0,
     K           3.5D0,   .8D0,             0.D0,0.D0,0.D0,0.D0,0.D0,
     L            .6D0,   .1D0,  -.5D0,   .8D0,          0.D0,0.D0,0.D0,
     M           -.8D0, -1.0D0,  1.4D0, -1.6D0,          0.D0,0.D0,0.D0,
     N           -.9D0,  -.8D0,  1.3D0, -1.6D0,          0.D0,0.D0,0.D0,
     O           3.5D0,   .8D0, -3.1D0,  4.8D0,          0.D0,0.D0,0.D0/
*                        TRUE Y RESULTS FOR ROTATIONS DROTM
      DATA DT19YA/.5D0,                  0.D0,0.D0,0.D0,0.D0,0.D0,0.D0,
     A            .5D0,                  0.D0,0.D0,0.D0,0.D0,0.D0,0.D0,
     B            .5D0,                  0.D0,0.D0,0.D0,0.D0,0.D0,0.D0,
     C            .5D0,                  0.D0,0.D0,0.D0,0.D0,0.D0,0.D0,
     D            .5D0,                  0.D0,0.D0,0.D0,0.D0,0.D0,0.D0,
     E            .7D0,                  0.D0,0.D0,0.D0,0.D0,0.D0,0.D0,
     F           1.7D0,                  0.D0,0.D0,0.D0,0.D0,0.D0,0.D0,
     G          -2.6D0,                  0.D0,0.D0,0.D0,0.D0,0.D0,0.D0,
     H            .5D0,  -.9D0,             0.D0,0.D0,0.D0,0.D0,0.D0,
     I            .7D0, -4.8D0,             0.D0,0.D0,0.D0,0.D0,0.D0,
     J           1.7D0,  -.7D0,             0.D0,0.D0,0.D0,0.D0,0.D0,
     K          -2.6D0,  3.5D0,             0.D0,0.D0,0.D0,0.D0,0.D0,
     L            .5D0,  -.9D0,   .3D0,   .7D0,          0.D0,0.D0,0.D0,
     M            .7D0, -4.8D0,  3.0D0,  1.1D0,          0.D0,0.D0,0.D0,
     N           1.7D0,  -.7D0,  -.7D0,  2.3D0,          0.D0,0.D0,0.D0,
     O          -2.6D0,  3.5D0,  -.7D0, -3.6D0,          0.D0,0.D0,0.D0/
*
      DATA DT19YB/.5D0,                  0.D0,0.D0,0.D0,0.D0,0.D0,0.D0,
     A            .5D0,                  0.D0,0.D0,0.D0,0.D0,0.D0,0.D0,
     B            .5D0,                  0.D0,0.D0,0.D0,0.D0,0.D0,0.D0,
     C            .5D0,                  0.D0,0.D0,0.D0,0.D0,0.D0,0.D0,
     D            .5D0,                  0.D0,0.D0,0.D0,0.D0,0.D0,0.D0,
     E            .7D0,                  0.D0,0.D0,0.D0,0.D0,0.D0,0.D0,
     F           1.7D0,                  0.D0,0.D0,0.D0,0.D0,0.D0,0.D0,
     G          -2.6D0,                  0.D0,0.D0,0.D0,0.D0,0.D0,0.D0,
     H            .5D0,  -.9D0,   .3D0,             0.D0,0.D0,0.D0,0.D0,
     I           4.0D0,  -.9D0,  -.3D0,             0.D0,0.D0,0.D0,0.D0,
     J           -.5D0,  -.9D0,  1.5D0,             0.D0,0.D0,0.D0,0.D0,
     K          -1.5D0,  -.9D0, -1.8D0,             0.D0,0.D0,0.D0,0.D0,
     L            .5D0,  -.9D0,   .3D0,   .7D0,  -.6D0,   .2D0,   .8D0,
     M           3.7D0,  -.9D0, -1.2D0,   .7D0, -1.5D0,   .2D0,  2.2D0,
     N           -.3D0,  -.9D0,  2.1D0,   .7D0, -1.6D0,   .2D0,  2.0D0,
     O          -1.6D0,  -.9D0, -2.1D0,   .7D0,  2.9D0,   .2D0, -3.8D0 /
*
      DATA DT19YC/.5D0,                  0.D0,0.D0,0.D0,0.D0,0.D0,0.D0,
     A            .5D0,                  0.D0,0.D0,0.D0,0.D0,0.D0,0.D0,
     B            .5D0,                  0.D0,0.D0,0.D0,0.D0,0.D0,0.D0,
     C            .5D0,                  0.D0,0.D0,0.D0,0.D0,0.D0,0.D0,
     D            .5D0,                  0.D0,0.D0,0.D0,0.D0,0.D0,0.D0,
     E            .7D0,                  0.D0,0.D0,0.D0,0.D0,0.D0,0.D0,
     F           1.7D0,                  0.D0,0.D0,0.D0,0.D0,0.D0,0.D0,
     G          -2.6D0,                  0.D0,0.D0,0.D0,0.D0,0.D0,0.D0,
     H            .5D0,  -.9D0,             0.D0,0.D0,0.D0,0.D0,0.D0,
     I           4.0D0, -6.3D0,             0.D0,0.D0,0.D0,0.D0,0.D0,
     J           -.5D0,   .3D0,             0.D0,0.D0,0.D0,0.D0,0.D0,
     K          -1.5D0,  3.0D0,             0.D0,0.D0,0.D0,0.D0,0.D0,
     L            .5D0,  -.9D0,   .3D0,   .7D0,          0.D0,0.D0,0.D0,
     M           3.7D0, -7.2D0,  3.0D0,  1.7D0,          0.D0,0.D0,0.D0,
     N           -.3D0,   .9D0,  -.7D0,  1.9D0,          0.D0,0.D0,0.D0,
     O          -1.6D0,  2.7D0,  -.7D0, -3.4D0,          0.D0,0.D0,0.D0/
*
      DATA DT19YD/.5D0,                  0.D0,0.D0,0.D0,0.D0,0.D0,0.D0,
     A            .5D0,                  0.D0,0.D0,0.D0,0.D0,0.D0,0.D0,
     B            .5D0,                  0.D0,0.D0,0.D0,0.D0,0.D0,0.D0,
     C            .5D0,                  0.D0,0.D0,0.D0,0.D0,0.D0,0.D0,
     D            .5D0,                  0.D0,0.D0,0.D0,0.D0,0.D0,0.D0,
     E            .7D0,                  0.D0,0.D0,0.D0,0.D0,0.D0,0.D0,
     F           1.7D0,                  0.D0,0.D0,0.D0,0.D0,0.D0,0.D0,
     G          -2.6D0,                  0.D0,0.D0,0.D0,0.D0,0.D0,0.D0,
     H            .5D0,  -.9D0,   .3D0,             0.D0,0.D0,0.D0,0.D0,
     I            .7D0,  -.9D0,  1.2D0,             0.D0,0.D0,0.D0,0.D0,
     J           1.7D0,  -.9D0,   .5D0,             0.D0,0.D0,0.D0,0.D0,
     K          -2.6D0,  -.9D0, -1.3D0,             0.D0,0.D0,0.D0,0.D0,
     L            .5D0,  -.9D0,   .3D0,   .7D0,  -.6D0,   .2D0,   .8D0,
     M            .7D0,  -.9D0,  1.2D0,   .7D0, -1.5D0,   .2D0,  1.6D0,
     N           1.7D0,  -.9D0,   .5D0,   .7D0, -1.6D0,   .2D0,  2.4D0,
     O          -2.6D0,  -.9D0, -1.3D0,   .7D0,  2.9D0,   .2D0, -4.0D0 /
*    
*     .. Executable Statements ..
*
      DO 120 KI = 1, 4
         INCX = INCXS(KI)
         INCY = INCYS(KI)
         MX = ABS(INCX)
         MY = ABS(INCY)
*
         DO 100 KN = 1, 4
            N = NS(KN)
            KSIZE = MIN(2,KN)
            LENX = LENS(KN,MX)
            LENY = LENS(KN,MY)
*           .. Initialize all argument arrays ..
            DO 20 I = 1, 7
               SX(I) = DX1(I)
               SY(I) = DY1(I)
   20       CONTINUE
*
            IF (ICASE.EQ.1) THEN
*              .. DDOT ..
               CALL STEST1(DDOT(N,SX,INCX,SY,INCY),DT7(KN,KI),SSIZE1(KN)
     +                     ,SFAC)
            ELSE IF (ICASE.EQ.2) THEN
*              .. DAXPY ..
               CALL DAXPY(N,SA,SX,INCX,SY,INCY)
               DO 40 J = 1, LENY
                  STY(J) = DT8(J,KN,KI)
   40          CONTINUE
               CALL STEST(LENY,SY,STY,SSIZE2(1,KSIZE),SFAC)
            ELSE IF (ICASE.EQ.5) THEN
*              .. DCOPY ..
               DO 60 I = 1, 7
                  STY(I) = DT10Y(I,KN,KI)
   60          CONTINUE
               CALL DCOPY(N,SX,INCX,SY,INCY)
               CALL STEST(LENY,SY,STY,SSIZE2(1,1),1.0D0)
            ELSE IF (ICASE.EQ.6) THEN
*              .. DSWAP ..
               CALL DSWAP(N,SX,INCX,SY,INCY)
               DO 80 I = 1, 7
                  STX(I) = DT10X(I,KN,KI)
               