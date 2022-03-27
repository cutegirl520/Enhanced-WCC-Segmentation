*> \brief \b SBLAT2
*
*  =========== DOCUMENTATION ===========
*
* Online html documentation available at 
*            http://www.netlib.org/lapack/explore-html/ 
*
*  Definition:
*  ===========
*
*       PROGRAM SBLAT2
* 
*
*> \par Purpose:
*  =============
*>
*> \verbatim
*>
*> Test program for the REAL Level 2 Blas.
*>
*> The program must be driven by a short data file. The first 18 records
*> of the file are read using list-directed input, the last 16 records
*> are read using the format ( A6, L2 ). An annotated example of a data
*> file can be obtained by deleting the first 3 characters from the
*> following 34 lines:
*> 'sblat2.out'      NAME OF SUMMARY OUTPUT FILE
*> 6                 UNIT NUMBER OF SUMMARY FILE
*> 'SBLAT2.SNAP'     NAME OF SNAPSHOT OUTPUT FILE
*> -1                UNIT NUMBER OF SNAPSHOT FILE (NOT USED IF .LT. 0)
*> F        LOGICAL FLAG, T TO REWIND SNAPSHOT FILE AFTER EACH RECORD.
*> F        LOGICAL FLAG, T TO STOP ON FAILURES.
*> T        LOGICAL FLAG, T TO TEST ERROR EXITS.
*> 16.0     THRESHOLD VALUE OF TEST RATIO
*> 6                 NUMBER OF VALUES OF N
*> 0 1 2 3 5 9       VALUES OF N
*> 4                 NUMBER OF VALUES OF K
*> 0 1 2 4           VALUES OF K
*> 4                 NUMBER OF VALUES OF INCX AND INCY
*> 1 2 -1 -2         VALUES OF INCX AND INCY
*> 3                 NUMBER OF VALUES OF ALPHA
*> 0.0 1.0 0.7       VALUES OF ALPHA
*> 3                 NUMBER OF VALUES OF BETA
*> 0.0 1.0 0.9       VALUES OF BETA
*> SGEMV  T PUT F FOR NO TEST. SAME COLUMNS.
*> SGBMV  T PUT F FOR NO TEST. SAME COLUMNS.
*> SSYMV  T PUT F FOR NO TEST. SAME COLUMNS.
*> SSBMV  T PUT F FOR NO TEST. SAME COLUMNS.
*> SSPMV  T PUT F FOR NO TEST. SAME COLUMNS.
*> STRMV  T PUT F FOR NO TEST. SAME COLUMNS.
*> STBMV  T PUT F FOR NO TEST. SAME COLUMNS.
*> STPMV  T PUT F FOR NO TEST. SAME COLUMNS.
*> STRSV  T PUT F FOR NO TEST. SAME COLUMNS.
*> STBSV  T PUT F FOR NO TEST. SAME COLUMNS.
*> STPSV  T PUT F FOR NO TEST. SAME COLUMNS.
*> SGER   T PUT F FOR NO TEST. SAME COLUMNS.
*> SSYR   T PUT F FOR NO TEST. SAME COLUMNS.
*> SSPR   T PUT F FOR NO TEST. SAME COLUMNS.
*> SSYR2  T PUT F FOR NO TEST. SAME COLUMNS.
*> SSPR2  T PUT F FOR NO TEST. SAME COLUMNS.
*>
*> Further Details
*> ===============
*>
*>    See:
*>
*>       Dongarra J. J., Du Croz J. J., Hammarling S.  and Hanson R. J..
*>       An  extended  set of Fortran  Basic Linear Algebra Subprograms.
*>
*>       Technical  Memoranda  Nos. 41 (revision 3) and 81,  Mathematics
*>       and  Computer Science  Division,  Argonne  National Laboratory,
*>       9700 South Cass Avenue, Argonne, Illinois 60439, US.
*>
*>       Or
*>
*>       NAG  Technical Reports TR3/87 and TR4/87,  Numerical Algorithms
*>       Group  Ltd.,  NAG  Central  Office,  256  Banbury  Road, Oxford
*>       OX2 7DE, UK,  and  Numerical Algorithms Group Inc.,  1101  31st
*>       Street,  Suite 100,  Downers Grove,  Illinois 60515-1263,  USA.
*>
*>
*> -- Written on 10-August-1987.
*>    Richard Hanson, Sandia National Labs.
*>    Jeremy Du Croz, NAG Central Office.
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
*> \ingroup single_blas_testing
*
*  =====================================================================
      PROGRAM SBLAT2
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
      PARAMETER          ( NSUBS = 16 )
      REAL               ZERO, ONE
      PARAMETER          ( ZERO = 0.0, ONE = 1.0 )
      INTEGER            NMAX, INCMAX
      PARAMETER          ( NMAX = 65, INCMAX = 2 )
      INTEGER            NINMAX, NIDMAX, NKBMAX, NALMAX, NBEMAX
      PARAMETER          ( NINMAX = 7, NIDMAX = 9, NKBMAX = 7,
     $                   NALMAX = 7, NBEMAX = 7 )
*     .. Local Scalars ..
      REAL               EPS, ERR, THRESH
      INTEGER            I, ISNUM, J, N, NALF, NBET, NIDIM, NINC, NKB,
     $                   NOUT, NTRA
      LOGICAL            FATAL, LTESTT, REWI, SAME, SFATAL, TRACE,
     $                   TSTERR
      CHARACTER*1        TRANS
      CHARACTER*6        SNAMET
      CHARACTER*32       SNAPS, SUMMRY
*     .. Local Arrays ..
      REAL               A( NMAX, NMAX ), AA( NMAX*NMAX ),
     $                   ALF( NALMAX ), AS( NMAX*NMAX ), BET( NBEMAX ),
     $                   G( NMAX ), X( NMAX ), XS( NMAX*INCMAX ),
     $                   XX( NMAX*INCMAX ), Y( NMAX ),
     $                   YS( NMAX*INCMAX ), YT( NMAX ),
     $                   YY( NMAX*INCMAX ), Z( 2*NMAX )
      INTEGER            IDIM( NIDMAX ), INC( NINMAX ), KB( NKBMAX )
      LOGICAL            LTEST( NSUBS )
      CHARACTER*6        SNAMES( NSUBS )
*     .. External Functions ..
      REAL               SDIFF
      LOGICAL            LSE
      EXTERNAL           SDIFF, LSE
*     .. External Subroutines ..
      EXTERNAL           SCHK1, SCHK2, SCHK3, SCHK4, SCHK5, SCHK6,
     $                   SCHKE, SMVCH
*     .. Intrinsic Functions ..
      INTRINSIC          ABS, MAX, MIN
*     .. Scalars in Common ..
      INTEGER            INFOT, NOUTC
      LOGICAL            LERR, OK
      CHARACTER*6        SRNAMT
*     .. Common blocks ..
      COMMON             /INFOC/INFOT, NOUTC, OK, LERR
      COMMON             /SRNAMC/SRNAMT
*     .. Data statements ..
      DATA               SNAMES/'SGEMV ', 'SGBMV ', 'SSYMV ', 'SSBMV ',
     $                   'SSPMV ', 'STRMV ', 'STBMV ', 'STPMV ',
     $                   'STRSV ', 'STBSV ', 'STPSV ', 'SGER  ',
     $                   'SSYR  ', 'SSPR  ', 'SSYR2 ', 'SSPR2 '/
*     .. Executable Statements ..
*
*     Read name and unit number for summary output file and open file.
*
      READ( NIN, FMT = * )SUMMRY
      READ( NIN, FMT = * )NOUT
      OPEN( NOUT, FILE = SUMMRY, STATUS = 'UNKNOWN' )
      NOUTC = NOUT
*
*     Read name and unit number for snapshot output file and open file.
*
      READ( NIN, FMT = * )SNAPS
      READ( NIN, FMT = * )NTRA
      TRACE = NTRA.GE.0
      IF( TRACE )THEN
         OPEN( NTRA, FILE = SNAPS, STATUS = 'UNKNOWN' )
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
         GO TO 230
      END IF
      READ( NIN, FMT = * )( IDIM( I ), I = 1, NIDIM )
      DO 10 I = 1, NIDIM
         IF( IDIM( I ).LT.0.OR.IDIM( I ).GT.NMAX )THEN
            WRITE( NOUT, FMT = 9996 )NMAX
            GO TO 230
         END IF
   10 CONTINUE
*     Values of K
      READ( NIN, FMT = * )NKB
      IF( NKB.LT.1.OR.NKB.GT.NKBMAX )THEN
         WRITE( NOUT, FMT = 9997 )'K', NKBMAX
         GO TO 230
      END IF
      READ( NIN, FMT = * )( KB( I ), I = 1, NKB )
      DO 20 I = 1, NKB
         IF( KB( I ).LT.0 )THEN
            WRITE( NOUT, FMT = 9995 )
            GO TO 230
         END IF
   20 CONTINUE
*     Values of INCX and INCY
      READ( NIN, FMT = * )NINC
      IF( NINC.LT.1.OR.NINC.GT.NINMAX )THEN
         WRITE( NOUT, FMT = 9997 )'INCX AND INCY', NINMAX
         GO TO 230
      END IF
      READ( NIN, FMT = * )( INC( I ), I = 1, NINC )
      DO 30 I = 1, NINC
         IF( INC( I ).EQ.0.OR.ABS( INC( I ) ).GT.INCMAX )THEN
            WRITE( NOUT, FMT = 9994 )INCMAX
            GO TO 230
         END IF
   30 CONTINUE
*     Values of ALPHA
      READ( NIN, FMT = * )NALF
      IF( NALF.LT.1.OR.NALF.GT.NALMAX )THEN
         WRITE( NOUT, FMT = 9997 )'ALPHA', NALMAX
         GO TO 230
      END IF
      READ( NIN, FMT = * )( ALF( I ), I = 1, NALF )
*     Values of BETA
      READ( NIN, FMT = * )NBET
      IF( NBET.LT.1.OR.NBET.GT.NBEMAX )THEN
         WRITE( NOUT, FMT = 9997 )'BETA', NBEMAX
         GO TO 230
      END IF
      READ( NIN, FMT = * )( BET( I ), I = 1, NBET )
*
*     Report values of parameters.
*
      WRITE( NOUT, FMT = 9993 )
      WRITE( NOUT, FMT = 9992 )( IDIM( I ), I = 1, NIDIM )
      WRITE( NOUT, FMT = 9991 )( KB( I ), I = 1, NKB )
      WRITE( NOUT, FMT = 9990 )( INC( I ), I = 1, NINC )
      WRITE( NOUT, FMT = 9989 )( ALF( I ), I = 1, NALF )
      WRITE( NOUT, FMT = 9988 )( BET( I ), I = 1, NBET )
      IF( .NOT.TSTERR )THEN
         WRITE( NOUT, FMT = * )
         WRITE( NOUT, FMT = 9980 )
      END IF
      WRITE( NOUT, FMT = * )
      WRITE( NOUT, FMT = 9999 )THRESH
      WRITE( NOUT, FMT = * )
*
*     Read names of subroutines and flags which indicate
*     whether they are to be tested.
*
      DO 40 I = 1, NSUBS
         LTEST( I ) = .FALSE.
   40 CONTINUE
   50 READ( NIN, FMT = 9984, END = 80 )SNAMET, LTESTT
      DO 60 I = 1, NSUBS
         IF( SNAMET.EQ.SNAMES( I ) )
     $      GO TO 70
   60 CONTINUE
      WRITE( NOUT, FMT = 9986 )SNAMET
      STOP
   70 LTEST( I ) = LTESTT
      GO TO 50
*
   80 CONTINUE
      CLOSE ( NIN )
*
*     Compute EPS (the machine precision).
*
      EPS = EPSILON(ZERO)
      WRITE( NOUT, FMT = 9998 )EPS
*
*     Check the reliability of SMVCH using exact data.
*
      N = MIN( 32, NMAX )
      DO 120 J = 1, N
         DO 110 I = 1, N
            A( I, J ) = MAX( I - J + 1, 0 )
  110    CONTINUE
         X( J ) = J
         Y( J ) = ZERO
  120 CONTINUE
      DO 130 J = 1, N
         YY( J ) = J*( ( J + 1 )*J )/2 - ( ( J + 1 )*J*( J - 1 ) )/3
  130 CONTINUE
*     YY holds the exact result. On exit from SMVCH YT holds
*     the result computed by SMVCH.
      TRANS = 'N'
      CALL SMVCH( TRANS, N, N, ONE, A, NMAX, X, 1, ZERO, Y, 1, YT, G,
     $            YY, EPS, ERR, FATAL, NOUT, .TRUE. )
      SAME = LSE( YY, YT, N )
      IF( .NOT.SAME.OR.ERR.NE.ZERO )THEN
         WRITE( NOUT, FMT = 9985 )TRANS, SAME, ERR
         STOP
      END IF
      TRANS = 'T'
      CALL SMVCH( TRANS, N, N, ONE, A, NMAX, X, -1, ZERO, Y, -1, YT, G,
     $            YY, EPS, ERR, FATAL, NOUT, .TRUE. )
      SAME = LSE( YY, YT, N )
      IF( .NOT.SAME.OR.ERR.NE.ZERO )THEN
         WRITE( NOUT, FMT = 9985 )TRANS, SAME, ERR
         STOP
      END IF
*
*     Test each subroutine in turn.
*
      DO 210 ISNUM = 1, NSUBS
         WRITE( NOUT, FMT = * )
         IF( .NOT.LTEST( ISNUM ) )THEN
*           Subprogram is not to be tested.
            WRITE( NOUT, FMT = 9983 )SNAMES( ISNUM )
         ELSE
            SRNAMT = SNAMES( ISNUM )
*           Test error exits.
            IF( TSTERR )THEN
               CALL SCHKE( ISNUM, SNAMES( ISNUM ), NOUT )
               WR