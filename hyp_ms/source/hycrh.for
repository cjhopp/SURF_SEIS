      SUBROUTINE HYCRH
C--READS A HOMOGENEOUS LAYER CRUSTAL MODEL FOR HYPOINVERSE
      INCLUDE 'common.inc'

      MODTYP(MOD)=1
      LAY(MOD)=0
      READ (14,1002,END=20,ERR=20) MODNAM(MOD)
1002  FORMAT (A)
      CRODE(MOD)=MODNAM(MOD)(1:3)

C--READ VELOCITY & DEPTH OF EACH LAYER
      DO 10 L=1,NLYR
      READ (14,1000,END=20) VEL(L,MOD),D(L,MOD)
1000  FORMAT (2F5.2)
C--COMPUTE THICKNESS & V**2 FOR LAYER
      LAY(MOD)=L
      IF (L.GT.1) THEN
        DD=D(L,MOD)-D(L-1,MOD)
        THK(L-1,MOD)=DD
        DV=VEL(L,MOD)-VEL(L-1,MOD)
        IF (DV.LT.0. .OR. DD.LE.0.) GOTO 22
      END IF
10    VSQ(L,MOD)=VEL(L,MOD)**2

C--DEFINE THK FOR HALFSPACE
20    THK(LAY(MOD),MOD)=999.
      RETURN

C--BAD DATA
22    WRITE (6,1001) MOD
1001  FORMAT (' *** BAD DATA FOR LAYER CRUST MODEL',I2)
      IRES=-95
      STOP
      END
