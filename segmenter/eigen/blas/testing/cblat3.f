*> \brief \b CBLAT3
*
*  =========== DOCUMENTATION ===========
*
* Online html documentation available at 
*            http://www.netlib.org/lapack/explore-html/ 
*
*  Definition:
*  ===========
*
*       PROGRAM CBLAT3
* 
*
*> \par Purpose:
*  =============
*>
*> \verbatim
*>
*> Test program for the COMPLEX          Level 3 Blas.
*>
*> The program must be driven by a short data file. The first 14 records
*> of the file are read using list-directed input, the last 9 records
*> are read using the format ( A6, L2 ). An annotated example of a data
*> file can be obtained by deleting the first 3 characters from the
*> following 23 lines:
*> 'cblat3.out'      NAME OF SUMMARY OUTPUT FILE
*> 6                 UNIT NUMBER OF SUMMARY FILE
*> 'CBLAT3.SNAP'     NAME OF SNAPSHOT OUTPUT FILE
*> -1                UNIT NUMBER OF SNAPSHOT FILE (NOT USED IF .LT. 0)
*> F        LOGICAL FLAG, T TO REWIND SNAPSHOT FILE AFTER EACH RECORD.
*> F        LOGICAL FLAG, T TO STOP ON FAILURES.
*> T        LOGICAL FLAG, T TO TEST ERROR EXITS.
*> 16.0     THRESHOLD VALUE OF TEST RATIO
*> 6                 NUMBER OF VALUES OF N
*> 0 1 2 3 5 9       VALUES OF N
*> 3                 NUMBER OF VALUES OF ALPHA
*> (0.0,0.0) (1.0,0.0) (0.7,-0.9)       VALUES OF ALPHA
*> 3                 NUMBER OF VALUES OF BETA
*> (0.0,0.0) (1.0,0.0) (1.3,-1.1)       VALUES OF BETA
*> CGEMM  T PUT F FOR NO TEST. SAME COLUMNS.
*> CHEMM  T PUT F FOR NO TEST. SAME COLUMNS.
*> CSYMM  T PUT F FOR NO TEST. SAME COLUMNS.
*> CTRMM  T PUT F FOR NO TEST. SAME COLUMNS.
*> CTRSM  T PUT F FOR NO TEST. SAME COLUMNS.
*> CHERK  T PUT F FOR NO TEST. SAME COLUMNS.
*> CSYRK  T PUT F FOR NO TEST. SAME COLUMNS.
*> CHER2K T PUT F FOR NO TEST. SAME COLUMNS.
*> CSYR2K T PUT F FOR NO TEST. SAME COLUMNS.
*>
*> Further Details
*> ===============
*>
*> See:
*>
*>    Dongarra J. J., Du Croz J. J., Duff I. S. and Hammarling S.
*>    A Set of Level 3 Basic Linear Algebra Subprograms.
*>
*>    Technical Memorandum No.88 (Revision 1), Mathematics and
*>    Computer Science Division, Argonne National Laboratory, 9700
*>    South Cass Avenue, Argonne, Illinois 60439, US.
*>
*> -- Written on 8-February-1989.
*>    Jack Dongarra, Argonne National Laboratory.
*>    Iain Duff, AERE Harwell.
*>    Jeremy Du Croz, Numerical Algorithms Group Ltd.
*>    Sven Hammarling, Numerical Algorithms Group Ltd.
*>
*>    10-9-00:  Change STATUS='NEW' to 'UNKNOWN' so that the testers
*>              can be run multiple times without deleting generated
*>              output files (susan)
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
*> \ingroup complex_blas_testing
*
*  =====================================================================
      PROGRAM CBLAT3
*
*  -- Reference BLAS test routine (version 3.4.1) --
*  -- Reference BLAS is a software package provided by Univ. of Tennessee,    --
*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
*     April 2012
*
*  =====================================================================
*
*     .. Parameters ..
      INTEGER            NIN
      PARAMETER          ( NIN = 5 )
      INTEGER            NSUBS
      PARAMETER          ( NSUBS = 9 )
      COMPLEX            ZERO, ONE
      PARAMETER          ( ZERO = ( 0.0, 0.0 ), ONE = ( 1.0, 0.0 ) )
      REAL               RZERO
      PARAMETER          ( RZERO = 0.0 )
      INTEGER            NMAX
      PARAMETER          ( NMAX = 65 )
      INTEGER            NIDMAX, NALMAX, NBEMAX
      PARAMETER          ( NIDMAX = 9, NALMAX = 7, NBEMAX = 7 )
*     .. Local Scalars ..
      REAL               EPS, ERR, THRESH
      INTEGER            I, ISNUM, J, N, NALF, NBET, NIDIM, NOUT, NTRA
      LOGICAL            FATAL, LTESTT, REWI, SAME, SFATAL, TRACE,
     $                   TSTERR
      CHARACTER*1        TRANSA, TRANSB
      CHARACTER*6        SNAMET
      CHARACTER*32       SNAPS, SUMMRY
*     .. Local Arrays ..
      COMPLEX            AA( NMAX*NMAX ), AB( NMAX, 2*NMAX ),
     $                   ALF( NALMAX ), AS( NMAX*NMAX ),
     $                   BB( NMAX*NMAX ), BET( NBEMAX ),
     $                   BS( NMAX*NMAX ), C( NMAX, NMAX ),
     $                   CC( NMAX*NMAX ), CS( NMAX*NMAX ), CT( NMAX ),
     $                   W( 2*NMAX )
      REAL               G( NMAX )
      INTEGER            IDIM( NIDMAX )
      LOGICAL            LTEST( NSUBS )
      CHARACTER*6        SNAMES( NSUBS )
*     .. External Functions ..
      REAL               SDIFF
      LOGICAL            LCE
      EXTERNAL           SDIFF, LCE
*     .. External Subroutines ..
      EXTERNAL           CCHK1, CCHK2, CCHK3, CCHK4, CCHK5, CCHKE, CMMCH
*     .. Intrinsic Functions ..
      INTRINSIC          MAX, MIN
*     .. Scalars in Common ..
      INTEGER            INFOT, NOUTC
      LOGICAL            LERR, OK
      CHARACTER*6        SRNAMT
*     .. Common blocks ..
      COMMON             /INFOC/INFOT, NOUTC, OK, LERR
      COMMON             /SRNAMC/SRNAMT
*     .. Data statements ..
      DATA               SNAMES/'CGEMM ', 'CHEMM ', 'CSYMM ', 'CTRMM ',
     $                   'CTRSM ', 'CHERK ', 'CSYRK ', 'CHER2K',
     $                   'CSYR2K'/
*     .. Executable Statements ..
*
*     Read name and unit number for summary output file and open file.
*
      READ( NIN, FMT = * )SUMMRY
      READ( NIN, FMT = * )NOUT
      OPEN( NOUT, FILE = SUMMRY )
      NOUTC = NOUT
*
*     Read name and unit number for snapshot output file and open file.
*
      READ( NIN, FMT = * )SNAPS
      READ( NIN, FMT = * )NTRA
      TRACE = NTRA.GE.0
      IF( TRACE )THEN
         OPEN( NTRA, FILE = SNAPS )
      END IF
*     Read the flag that directs rewinding of the snapshot file.
      READ( NIN, FMT = * )REWI
      REWI = REWI.AND.TRACE
*     Read the flag that directs stopping on any failure.
      READ( NIN, FMT = * )SFATAL
*     Read the flag that indicates whether error exits are to be tested.
      READ( NIN, FMT = * )TSTERR
*     Read the threshold value of the test ratio
      READ( NIN, FMT = * )THRESH
*
*     Read and check the parameter values for the tests.
*
*     Values of N
      READ( NIN, FMT = * )NIDIM
      IF( NIDIM.LT.1.OR.NIDIM.GT.NIDMAX )THEN
         WRITE( NOUT, FMT = 9997 )'N', NIDMAX
         GO TO 220
      END IF
      READ( NIN, FMT = * )( IDIM( I ), I = 1, NIDIM )
      DO 10 I = 1, NIDIM
         IF( IDIM( I ).LT.0.OR.IDIM( I ).GT.NMAX )THEN
            WRITE( NOUT, FMT = 9996 )NMAX
            GO TO 220
         END IF
   10 CONTINUE
*     Values of ALPHA
      READ( NIN, FMT = * )NALF
      IF( NALF.LT.1.OR.NALF.GT.NALMAX )THEN
         WRITE( NOUT, FMT = 9997 )'ALPHA', NALMAX
         GO TO 220
      END IF
      READ( NIN, FMT = * )( ALF( I ), I = 1, NALF )
*     Values of BETA
      READ( NIN, FMT = * )NBET
      IF( NBET.LT.1.OR.NBET.GT.NBEMAX )THEN
         WRITE( NOUT, FMT = 9997 )'BETA', NBEMAX
         GO TO 220
      END IF
      READ( NIN, FMT = * )( BET( I ), I = 1, NBET )
*
*     Report values of parameters.
*
      WRITE( NOUT, FMT = 9995 )
      WRITE( NOUT, FMT = 9994 )( IDIM( I ), I = 1, NIDIM )
      WRITE( NOUT, FMT = 9993 )( ALF( I ), I = 1, NALF )
      WRITE( NOUT, FMT = 9992 )( BET( I ), I = 1, NBET )
      IF( .NOT.TSTERR )THEN
         WRITE( NOUT, FMT = * )
         WRITE( NOUT, FMT = 9984 )
      END IF
      WRITE( NOUT, FMT = * )
      WRITE( NOUT, FMT = 9999 )THRESH
      WRITE( NOUT, FMT = * )
*
*     Read names of subroutines and flags which indicate
*     whether they are to be tested.
*
      DO 20 I = 1, NSUBS
         LTEST( I ) = .FALSE.
   20 CONTINUE
   30 READ( NIN, FMT = 9988, END = 60 )SNAMET, LTESTT
      DO 40 I = 1, NSUBS
         IF( SNAMET.EQ.SNAMES( I ) )
     $      GO TO 50
   40 CONTINUE
      WRITE( NOUT, FMT = 9990 )SNAMET
      STOP
   50 LTEST( I ) = LTESTT
      GO TO 30
*
   60 CONTINUE
      CLOSE ( NIN )
*
*     Compute EPS (the machine precision).
*
      EPS = EPSILON(RZERO)
      WRITE( NOUT, FMT = 9998 )EPS
*
*     Check the reliability of CMMCH using exact data.
*
      N = MIN( 32, NMAX )
      DO 100 J = 1, N
         DO 90 I = 1, N
            AB( I, J ) = MAX( I - J + 1, 0 )
   90    CONTINUE
         AB( J, NMAX + 1 ) = J
         AB( 1, NMAX + J ) = J
         C( J, 1 ) = ZERO
  100 CONTINUE
      DO 110 J = 1, N
         CC( J ) = J*( ( J + 1 )*J )/2 - ( ( J + 1 )*J*( J - 1 ) )/3
  110 CONTINUE
*     CC holds the exact result. On exit from CMMCH CT holds
*     the result computed by CMMCH.
      TRANSA = 'N'
      TRANSB = 'N'
      CALL CMMCH( TRANSA, TRANSB, N, 1, N, ONE, AB, NMAX,
     $            AB( 1, NMAX + 1 ), NMAX, ZERO, C, NMAX, CT, G, CC,
     $            NMAX, EPS, ERR, FATAL, NOUT, .TRUE. )
      SAME = LCE( CC, CT, N )
      IF( .NOT.SAME.OR.ERR.NE.RZERO )THEN
         WRITE( NOUT, FMT = 9989 )TRANSA, TRANSB, SAME, ERR
         STOP
      END IF
      TRANSB = 'C'
      CALL CMMCH( TRANSA, TRANSB, N, 1, N, ONE, AB, NMAX,
     $            AB( 1, NMAX + 1 ), NMAX, ZERO, C, NMAX, CT, G, CC,
     $            NMAX, EPS, ERR, FATAL, NOUT, .TRUE. )
      SAME = LCE( CC, CT, N )
      IF( .NOT.SAME.OR.ERR.NE.RZERO )THEN
         WRITE( NOUT, FMT = 9989 )TRANSA, TRANSB, SAME, ERR
         STOP
      END IF
      DO 120 J = 1, N
         AB( J, NMAX + 1 ) = N - J + 1
         AB( 1, NMAX + J ) = N - J + 1
  120 CONTINUE
      DO 130 J = 1, N
         CC( N - J + 1 ) = J*( ( J + 1 )*J )/2 -
     $                     ( ( J + 1 )*J*( J - 1 ) )/3
  130 CONTINUE
      TRANSA = 'C'
      TRANSB = 'N'
      CALL CMMCH( TRANSA, TRANSB, N, 1, N, ONE, AB, NMAX,
     $            AB( 1, NMAX + 1 ), NMAX, ZERO, C, NMAX, CT, G, CC,
     $            NMAX, EPS, ERR, FATAL, NOUT, .TRUE. )
      SAME = LCE( CC, CT, N )
      IF( .NOT.SAME.OR.ERR.NE.RZERO )THEN
         WRITE( NOUT, FMT = 9989 )TRANSA, TRANSB, SAME, ERR
         STOP
      END IF
      TRANSB = 'C'
      CALL CMMCH( TRANSA, TRANSB, N, 1, N, ONE, AB, NMAX,
     $            AB( 1, NMAX + 1 ), NMAX, ZERO, C, NMAX, CT, G, CC,
     $            NMAX, EPS, ERR, FATAL, NOUT, .TRUE. )
      SAME = LCE( CC, CT, N )
      IF( .NOT.SAME.OR.ERR.NE.RZERO )THEN
         WRITE( NOUT, FMT = 9989 )TRANSA, TRANSB, SAME, ERR
         STOP
      END IF
*
*     Test each subroutine in turn.
*
      DO 200 ISNUM = 1, NSUBS
         WRITE( NOUT, FMT = * )
         IF( .NOT.LTEST( ISNUM ) )THEN
*           Subprogram is not to be tested.
            WRITE( NOUT, FMT = 9987 )SNAMES( ISNUM )
         ELSE
            SRNAMT = SNAMES( ISNUM )
*           Test error exits.
            IF( TSTERR )THEN
               CALL CCHKE( ISNUM, SNAMES( ISNUM ), NOUT )
               WRITE( NOUT, FMT = * )
            END IF
*           Test computations.
            INFOT = 0
            OK = .TRUE.
            FATAL = .FALSE.
            GO TO ( 140, 150, 150, 160, 160, 170, 170,
     $              180, 180 )ISNUM
*           Test CGEMM, 01.
  140       CALL CCHK1( SNAMES( ISNUM ), EPS, THRESH, NOUT, NTRA, TRACE,
     $                  REWI, FATAL, NIDIM, IDIM, NALF, ALF, NBET, BET,
     $                  NMAX, AB, AA, AS, AB( 1, NMAX + 1 ), BB, BS, C,
     $                  CC, CS, CT, G )
            GO TO 190
*           Test CHEMM, 02, CSYMM, 03.
  150       CALL CCHK2( SNAMES( ISNUM ), EPS, THRESH, NOUT, NTRA, TRACE,
     $                  REWI, FATAL, NIDIM, IDIM, NALF, ALF, NBET, BET,
     $                  NMAX, AB, AA, AS, AB( 1, NMAX + 1 ), BB, BS, C,
     $                  CC, CS, CT, G )
            GO TO 190
*           Test CTRMM, 04, CTRSM, 05.
  160       CALL CCH