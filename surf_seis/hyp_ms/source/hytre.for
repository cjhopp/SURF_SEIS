      SUBROUTINE HYTRE
C--GIVEN DEPTH & DISTANCE, THIS ROUTINE CALCULATES TRAVEL TIME, ITS
C--DERIVATIVES AND EMERGENCE ANGLES AT THE SOURCE FOR ALL ARRIVALS.
C  USES THE TRVCON LAYER MODEL CALCULATOR FROM HYPOELLIPSE
      INCLUDE 'common.inc'
      LOGICAL ALTMOD, SALMOD
      DIMENSION VNOW(NLYR),THKNOW(NLYR),V2NOW(NLYR)	!CURRENT MODEL
      DIMENSION VINOW(NLYR)
      CHARACTER MSTA*5				!?

C--ARRAYS FOR TRVCON ARGUMENTS (USED BY HYPOELLIPSE)
      DIMENSION TIDNOW (NLYR,NLYR), DIDNOW (NLYR,NLYR)
      DIMENSION VSQDNOW (NLYR,NLYR), FNOW (NLYR,NLYR)

C--JREF LABELS LAYERS WHICH CAN BE REFRACTORS (=1) AND TOP LAYER WHICH CANT(=0)
      DIMENSION JREF(NLYR)	!LABELS REFRACTING LAYERS
      JREF(1)=0
      DO I=2,NLYR
        JREF(I)=1
      END DO

C--THE FOLLOWING ARE PASSED THRU THE ARRAY A:
C DTDR      !TT DERIV WRT DISTANCE
C DTDZ      !TT DERIV WRT DEPTH
C T      !TRAVEL TIME
C AIN      !ANGLE OF EMERGENCE AT SOURCE

C--STILL NEED MODEL RENUMBERING CODE
      ALTMOD=MODALT(MOD).GT.0
      SALMOD=MODSAL(MOD).GT.0

C--LOOP OVER ALL ARRIVALS
      DO 280 I=1,M
C--FIND STATION INDEX AND REMOVE KPS AS AN S FLAG
      KI=IND(I)
      KPS=KI/10000
      KI=KI-10000*KPS
      J=KINDX(KI)

C--DETERMINE THE MODEL NO. TO ACTUALLY USE FOR THIS STATION
      MD=MOD
      IF (ALTMOD .AND. JLMOD(J)) MD=MODALT(MOD)
C--SWITCH TO S MODEL
      MDS=MD
      IF (SALMOD .AND. KPS.EQ.1) MD=MODSAL(MDS)

C--ZTM IS THE EQ DEPTH IN KM BELOW THE MODEL TOP (REFERENCE ELEVATION)
C  Z1 IS RELATIVE TO SEA LEVEL
      ZTM=Z1 +ELEVMX(MD)

C--PREPARE VELOCITY MODEL INFO IN 1D & 2D ARRAYS FROM COMMON AREA
      LAYNOW=LAY(MD)
      DO IL=1,LAYNOW
        VNOW(IL)=VEL(IL,MD)
        V2NOW(IL)=VSQ(IL,MD)
        THKNOW(IL)=THK(IL,MD)
        VINOW(IL)=VELI(IL,MD)
C--THESE 3D ARRAYS WERE DEFINED IN HYCRE AS MODEL WAS READ IN
        DO L=1,LAYNOW
          TIDNOW(IL,L)=TIDE(IL,L,MD)
          DIDNOW(IL,L)=DIDE(IL,L,MD)
          VSQDNOW(IL,L)=VSQDE(IL,L,MD)
          FNOW(IL,L)=FREF(IL,L,MD)
        END DO
      END DO

C--STATION DISTANCE
      DX=DIS(KI)
C--STZ IS THE STATION DEPTH IN KM BELOW THE REFERENCE ELEVATION
      STZ=ELEVMX(MD)
      IF (LELEV(MD)) STZ=ELEVMX(MD) -0.001*JELEV(J)
C--STATION CODE
      MSTA=STANAM(J)
      
C--ASSUME THE TOP OF THE MODEL SURFACE IS AT ELEVMX
C  (THE REFERENCE ELEVATION)

C--CALCULATE TRAVEL TIME & DERIVATIVES
C--VST=VELOCITY AT STATION, VEQ=VEL AT EQ (NOT USED)
C      CALL LINV(DX,Z1,VREF(MD),VGRAD(MD),T,AIN,DTDR,DTDZ,STZ,VST,VEQ)

C--THIS IS THE CALLING LIST FROM TRVDRV IN HYPOELLIPSE
C        call trvcon( delta(i), z, t(i), ain(i), dtdd, dtdh,
C     *            lbeg(imod), lend(imod), lbeg(imod)-1, nlayers,
C     *            ivlr, ivl, thk, nlay(i), ldx(i), wt(i), wtk,
C     *            tid, did, jref, vsq, vsqd, v, vi, f,
C     *            vs, vt, msta(i), stz)

C--THIS IS THE ARGUMENT LIST FROM TRVCON IN HYPOELLIPSE & HYPOINVERSE
C      subroutine trvcon(delta, zsv, t, ain, dtdd, dtdh,
C     *  lbeg, lend, lbegm1, nlayers,
C     *  ivlr, ivl, thk, nlay, ldx, wti, wtk,
C     *  tid, did, jref, vsq, vsqd, v, vi, f, vs, vt, msta, stzsv)

      CALL TRVCON (DX,	!EPICENTRAL DISTANCE IN KM
     * ZTM,		!DEPTH OF EQ BELOW MODEL TOP (REFERENCE ELEV)
     * T,		!CALCULATED TRAVEL TIME (RETURN FROM TRVCON)
     * AIN,		!CALCULATED ANGLE OF EMERGENCE AT SOURCE
     * DTDR,		!CALCULATED TT DERIV WRT DISTANCE
     * DTDZ,		!CALCULATED TT DERIV WRT DEPTH

     * 1,		!INDEX OF FIRST LAYER (LBEG)
     * LAYNOW,		!INDEX OF LAST LAYER (LEND)
     * 0,		!(LBEGM1) (LBEG-1)
     * LAYNOW,		!NUMBER OF LAYERS (NLAYERS)

     * 0,		!NO LAYER WITH VARIABLE THICKNESS (IVLR)
     * 0,		!NO LAYER WITH VARIABLE THICKNESS (IVL)
     * THKNOW,		!THICKNESS OF EACH LAYER OF CURRENT MODEL
     * 0,		!NO LAYER WITH A FORCED REFRACTION (NLAY) 
C-- (NLAY IS INPUT FROM HYPOE PHASE CARDS & INDICATES FORCED REFRACTION LAYER)
     * 0,		!NO HYPOELLIPSE LAYERED S MODELS (LDX)
C--THE FOLLOWING 2 VARIABLES ARE SET IN TRVCON BUT NEVER USED
     * WTI,		!WEIGHT (SET TO 0) FOR IMPOSSIBLE RAYS (NOT IMPLEMENTED)
     * WTK,		!WEIGHT (SET TO 0) FOR IMPOSSIBLE RAYS (NOT IMPLEMENTED)

     * TIDNOW,		!PREDEFINED ARRAYS RELATED TO REFRACTION
     * DIDNOW,		!PREDEFINED ARRAYS RELATED TO REFRACTION
     * JREF,		!RELATED TO REFRACTING LAYERS
     * V2NOW,		!VELOCITIES SQUARED OF LAYERS OF CURRENT MODEL
     * VSQDNOW,		!PREDEFINED ARRAYS RELATED TO REFRACTION

     * VNOW,		!VELOCITIES OF LAYERS OF CURRENT MODEL
     * VINOW,		!PREDEFINED ARRAYS RELATED TO REFRACTION (1/V)
     * FNOW,		!PREDEFINED ARRAYS RELATED TO REFRACTION
     * VSHYPO,		!CALCULATED VELOCITY AT HYPOCENTER
     * VSSTA,		!CALCULATED VELOCITY AT STATION
     * MSTA,		!STATION CODE FOR ERROR MESSAGE
     * STZ)		!STATION DEPTH (KM) BELOW TOP OF MODEL

C--END OF STATION LOOP
C--ASSIGN CALCULATIONS INTO ARRAY FOR INVERSION
      A(I,1)=AIN
      A(I,2)=T
      A(I,3)=DTDR
280   A(I,4)=DTDZ

      RETURN
      END
