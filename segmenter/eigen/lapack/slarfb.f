*> \brief \b SLARFB
*
*  =========== DOCUMENTATION ===========
*
* Online html documentation available at 
*            http://www.netlib.org/lapack/explore-html/ 
*
*> \htmlonly
*> Download SLARFB + dependencies 
*> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/slarfb.f"> 
*> [TGZ]</a> 
*> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/slarfb.f"> 
*> [ZIP]</a> 
*> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/slarfb.f"> 
*> [TXT]</a>
*> \endhtmlonly 
*
*  Definition:
*  ===========
*
*       SUBROUTINE SLARFB( SIDE, TRANS, DIRECT, STOREV, M, N, K, V, LDV,
*                          T, LDT, C, LDC, WORK, LDWORK )
* 
*       .. Scalar Arguments ..
*       CHARACTER          DIRECT, SIDE, STOREV, TRANS
*       INTEGER            K, LDC, LDT, LDV, LDWORK, M, N
*       ..
*       .. Array Arguments ..
*       REAL               C( LDC, * ), T( LDT, * ), V( LDV, * ),
*      $                   WORK( LDWORK, * )
*       ..
*  
*
*> \par Purpose:
*  =============
*>
*> \verbatim
*>
*> SLARFB applies a real block reflector H or its transpose H**T to a
*> real m by n matrix C, from either the left or the right.
*> \endverbatim
*
*  Arguments:
*  ==========
*
*> \param[in] SIDE
*> \verbatim
*>          SIDE is CHARACTER*1
*>          = 'L': apply H or H**T from the Left
*>          = 'R': apply H or H**T from the Right
*> \endverbatim
*>
*> \param[in] TRANS
*> \verbatim
*>          TRANS is CHARACTER*1
*>          = 'N': apply H (No transpose)
*>          = 'T': apply H**T (Transpose)
*> \endverbatim
*>
*> \param[in] DIRECT
*> \verbatim
*>          DIRECT is CHARACTER*1
*>          Indicates how H is formed from a product of elementary
*>          reflectors
*>          = 'F': H = H(1) H(2) . . . H(k) (Forward)
*>          = 'B': H = H(k) . . . H(2) H(1) (Backward)
*> \endverbatim
*>
*> \param[in] STOREV
*> \verbatim
*>          STOREV is CHARACTER*1
*>          Indicates how the vectors which define the elementary
*>          reflectors are stored:
*>          = 'C': Columnwise
*>          = 'R': Rowwise
*> \endverbatim
*>
*> \param[in] M
*> \verbatim
*>          M is INTEGER
*>          The number of rows of the matrix C.
*> \endverbatim
*>
*> \param[in] N
*> \verbatim
*>          N is INTEGER
*>          The number of columns of the matrix C.
*> \endverbatim
*>
*> \param[in] K
*> \verbatim
*>          K is INTEGER
*>          The order of the matrix T (= the number of elementary
*>          reflectors whose product defines the block reflector).
*> \endverbatim
*>
*> \param[in] V
*> \verbatim
*>          V is REAL array, dimension
*>                                (LDV,K) if STOREV = 'C'
*>                                (LDV,M) if STOREV = 'R' and SIDE = 'L'
*>                                (LDV,N) if STOREV = 'R' and SIDE = 'R'
*>          The matrix V. See Further Details.
*> \endverbatim
*>
*> \param[in] LDV
*> \verbatim
*>          LDV is INTEGER
*>          The leading dimension of the array V.
*>          If STOREV = 'C' and SIDE = 'L', LDV >= max(1,M);
*>          if STOREV = 'C' and SIDE = 'R', LDV >= max(1,N);
*>          if STOREV = 'R', LDV >= K.
*> \endverbatim
*>
*> \param[in] T
*> \verbatim
*>          T is REAL array, dimension (LDT,K)
*>          The triangular k by k matrix T in the representation of the
*>          block reflector.
*> \endverbatim
*>
*> \param[in] LDT
*> \verbatim
*>          LDT is INTEGER
*>          The leading dimension of the array T. LDT >= K.
*> \endverbatim
*>
*> \param[in,out] C
*> \verbatim
*>          C is REAL array, dimension (LDC,N)
*>          On entry, the m by n matrix C.
*>          On exit, C is overwritten by H*C or H**T*C or C*H or C*H**T.
*> \endverbatim
*>
*> \param[in] LDC
*> \verbatim
*>          LDC is INTEGER
*>          The leading dimension of the array C. LDC >= max(1,M).
*> \endverbatim
*>
*> \param[out] WORK
*> \verbatim
*>          WORK is REAL array, dimension (LDWORK,K)
*> \endverbatim
*>
*> \param[in] LDWORK
*> \verbatim
*>          LDWORK is INTEGER
*>          The leading dimension of the array WORK.
*>          If SIDE = 'L', LDWORK >= max(1,N);
*>          if SIDE = 'R', LDWORK >= max(1,M).
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
*> \date November 2011
*
*> \ingroup realOTHERauxiliary
*
*> \par Further Details:
*  =====================
*>
*> \verbatim
*>
*>  The shape of the matrix V and the storage of the vectors which define
*>  the H(i) is best illustrated by the following example with n = 5 and
*>  k = 3. The elements equal to 1 are not stored; the corresponding
*>  array elements are modified but restored on exit. The rest of the
*>  array is not used.
*>
*>  DIRECT = 'F' and STOREV = 'C':         DIRECT = 'F' and STOREV = 'R':
*>
*>               V = (  1       )                 V = (  1 v1 v1 v1 v1 )
*>                   ( v1  1    )                     (     1 v2 v2 v2 )
*>                   ( v1 v2  1 )                     (        1 v3 v3 )
*>                   ( v1 v2 v3 )
*>                   ( v1 v2 v3 )
*>
*>  DIRECT = 'B' and STOREV = 'C':         DIRECT = 'B' and STOREV = 'R':
*>
*>               V = ( v1 v2 v3 )                 V = ( v1 v1  1       )
*>                   ( v1 v2 v3 )                     ( v2 v2 v2  1    )
*>                   (  1 v2 v3 )                     ( v3 v3 v3 v3  1 )
*>                   (     1 v3 )
*>                   (        1 )
*> \endverbatim
*>
*  =====================================================================
      SUBROUTINE SLARFB( SIDE, TRANS, DIRECT, STOREV, M, N, K, V, LDV,
     $                   T, LDT, C, LDC, WORK, LDWORK )
*
*  -- LAPACK auxiliary routine (version 3.4.0) --
*  -- LAPACK is a software package provided by Univ. of Tennessee,    --
*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
*     November 2011
*
*     .. Scalar Arguments ..
      CHARACTER          DIRECT, SIDE, STOREV, TRANS
      INTEGER            K, LDC, LDT, LDV, LDWORK, M, N
*     ..
*     .. Array Arguments ..
      REAL               C( LDC, * ), T( LDT, * ), V( LDV, * ),
     $                   WORK( LDWORK, * )
*     ..
*
*  =====================================================================
*
*     .. Parameters ..
      REAL               ONE
      PARAMETER          ( ONE = 1.0E+0 )
*     ..
*     .. Local Scalars ..
      CHARACTER          TRANST
      INTEGER            I, J, LASTV, LASTC
*     ..
*     .. External Functions ..
      LOGICAL            LSAME
      INTEGER            ILASLR, ILASLC
      EXTERNAL           LSAME, ILASLR, ILASLC
*     ..
*     .. External Subroutines ..
      EXTERNAL           SCOPY, SGEMM, STRMM
*     ..
*     .. Executable Statements ..
*
*     Quick return if possible
*
      IF( M.LE.0 .OR. N.LE.0 )
     $   RETURN
*
      IF( LSAME( TRANS, 'N' ) ) THEN
         TRANST = 'T'
      ELSE
         TRANST = 'N'
      END IF
*
      IF( LSAME( STOREV, 'C' ) ) THEN
*
         IF( LSAME( DIRECT, 'F' ) ) THEN
*
*           Let  V =  ( V1 )    (first K rows)
*                     ( V2 )
*           where  V1  is unit lower triangular.
*
            IF( LSAME( SIDE, 'L' ) ) THEN
*
*              Form  H * C  or  H**T * C  where  C = ( C1 )
*                                                    ( C2 )
*
               LASTV = MAX( K, ILASLR( M, K, V, LDV ) )
               LASTC = ILASLC( LASTV, N, C, LDC )
*
*              W := C**T * V  =  (C1**T * V1 + C2**T * V2)  (stored in WORK)
*
*              W := C1**T
*
               DO 10 J = 1, K
                  CALL SCOPY( LASTC, C( J, 1 ), LDC, WORK( 1, J ), 1 )
   10          CONTINUE
*
*              W := W * V1
*
               CALL STRMM( 'Right', 'Lower', 'No transpose', 'Unit',
     $              LASTC, K, ONE, V, LDV, WORK, LDWORK )
               IF( LASTV.GT.K ) THEN
*
*                 W := W + C2**T *V2
*
                  CALL SGEMM( 'Transpose', 'No transpose',
     $                 LASTC, K, LASTV-K,
     $                 ONE, C( K+1, 1 ), LDC, V( K+1, 1 ), LDV,
     $                 ONE, WORK, LDWORK )
               END IF
*
*              W := W * T**T  or  W * T
*
               CALL STRMM( 'Right', 'Upper', TRANST, 'Non-unit',
     $              LASTC, K, ONE, T, LDT, WORK, LDWORK )
*
*              C := C - V * W**T
*
               IF( LASTV.GT.K ) THEN
*
*                 C2 := C2 - V2 * W**T
*
                  CALL SGEMM( 'No transpose', 'Transpose',
     $                 LASTV-K, LASTC, K,
     $                 -ONE, V( K+1, 1 ), LDV, WORK, LDWORK, ONE,
     $                 C( K+1, 1 ), LDC )
               END IF
*
*              W := W * V1**T
*
               CALL STRMM( 'Right', 'Lower', 'Transpose', 'Unit',
     $              LASTC, K, ONE, V, LDV, WORK, LDWORK )
*
*              C1 := C1 - W**T
*
               DO 30 J = 1, K
                  DO 20 I = 1, LASTC
                     C( J, I ) = C( J, I ) - WORK( I, J )
   20             CONTINUE
   30          CONTINUE
*
            ELSE IF( LSAME( SIDE, 'R' ) ) THEN
*
*              Form  C * H  or  C * H**T  where  C = ( C1  C2 )
*
               LASTV = MAX( K, ILASLR( N, K, V, LDV ) )
               LASTC = ILASLR( M, LASTV, C, LDC )
*
*              W := C * V  =  (C1*V1 + C2*V2)  (stored in WORK)
*
*              W := C1
*
               DO 40 J = 1, K
                  CALL SCOPY( LASTC, C( 1, J ), 1, WORK( 1, J ), 1 )
   40          CONTINUE
*
*              W := W * V1
*
               CALL STRMM( 'Right', 'Lower', 'No transpose', 'Unit',
     $              LASTC, K, ONE, V, LDV, WORK, LDWORK )
               IF( LASTV.GT.K ) THEN
*
*                 W := W + C2 * V2
*
