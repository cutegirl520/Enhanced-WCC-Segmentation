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
      DTRUE(1,6) = 4.D10 /