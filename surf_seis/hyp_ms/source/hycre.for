      SUBROUTINE HYCRE
C--READS A HYPOELLIPSE LAYER CRUSTAL MODEL FOR HYPOINVERSE
      INCLUDE 'common.inc'

      MODTYP(MOD)=4
      LAY(MOD)=0
      READ (14,1002,END=20,ERR=20) MODNAM(MOD)
1002  FORMAT (A)
      CRODE(MOD)=MODNAM(MOD)(1:3)

C--READ VELOCITY & DEPTH OF EACH LAYER
      DO L=1,NLYR
        READ (14,1000,END=20) VEL(L,MOD),D(L,MOD)
1000    FORMAT (2F5.2)
C--COMPUTE THICKNESS & V**2 FOR LAYER
        LAY(MOD)=L
        IF (L.GT.1) THEN
          DD=D(L,MOD)-D(L-1,MOD)
          THK(L-1,MOD)=DD
          DV=VEL(L,MOD)-VEL(L-1,MOD)
c          IF (DV.LT.0. .OR. DD.LE.0.) GOTO 22	!version 1.39 change
          IF (DD.LE.0.) GOTO 22
        END IF
        VSQ(L,MOD)=VEL(L,MOD)**2
        VELI(L,MOD)=1./VEL(L,MOD)
      END DO

C--DEFINE THK FOR HALFSPACE
20    THK(LAY(MOD),MOD)=999.

C--COMPUTE ARRAYS FOR REFRACTION CALCULATIONS (FROM HYPOEL INPUT SUB)
C  DO THIS FOR THE MODEL MOD
      CALL HYCRE2
      RETURN

C--BAD DATA
22    WRITE (6,1001) MOD
1001  FORMAT (' *** BAD DATA FOR LAYER CRUST MODEL',I2)
      IRES=-95
      STOP
      END

      SUBROUTINE HYCRE2
C--COMPUTE ARRAYS FOR REFRACTION CALCULATIONS (FROM HYPOEL INPUT SUB)
C  DO THIS FOR THE MODEL MOD, GOTTEN FROM THE COMMON AREA
      INCLUDE 'common.inc'

C--L IS THE LAYER, MREF IS THE REFRACTOR LAYER
      DO L=1,LAY(MOD)
        DO MREF=1,LAY(MOD)
          VRAT=VEL(MREF,MOD)/VEL(L,MOD)
          IF (MREF.GT.L .AND. VRAT.GT.1.) THEN
            VSQDE(MREF,L,MOD)=SQRT((VRAT-1.)*(VRAT+1.))
          ELSE
            VSQDE(MREF,L,MOD)=0.
          END IF
C--NOTE SUBSCRIPT ORDER REVERSES
          IF (L.GE.MREF) THEN
            FREF(L,MREF,MOD)=2.
          ELSE
            FREF(L,MREF,MOD)=1.
          END IF
        END DO
      END DO

C--COMPUTE MORE ARRAYS FOR REFRACTION CALCULATIONS (FROM HYPOEL INPUT SUB)
      DO L=1,LAY(MOD)
        DO MREF=L,LAY(MOD)
          IF (MREF.GT.1) THEN
            SUMT=0.
            SUMD=0.
C--LOOP FROM TOP LAYER TO LAYER ABOVE REFRACTOR
C--SKIP IF REFRACTOR VELOCITY LE ANY OVERLYING VELOCITY
            DO I=1,MREF-1
              IF (VEL(MREF,MOD) .LE. VEL(I,MOD)) GOTO 60
            END DO
C--THIS STATEMENT LABELS LAYERS CAPABLE OF BEING REFRACTORS, ALLOWING FOR LOW
C  VEL ZONES. BECAUSE HI DOES NOT HAVE LVZS, ALL LAYERS EX #1 CAN BE REFRACTORS.
C--SET JREF IN HYTRE.FOR WITH A DATA STATEMENT.
C            jref(mrefr + lbeg(imod) - 1) = 1

            DO I=1,MREF-1
              SUMT=SUMT +FREF(I,L,MOD)*THK(I,MOD)*VSQDE(MREF,I,MOD)
              SUMD=SUMD +FREF(I,L,MOD)*THK(I,MOD)/VSQDE(MREF,I,MOD)
            END DO
60          TIDE(L,MREF,MOD)=SUMT*VELI(MREF,MOD)
            DIDE(L,MREF,MOD)=SUMD
          END IF
        END DO
      END DO
      RETURN
      END
c--hypoellipse notes
c  lmax is the total number of layers of all models
c  lmmax is the max no of layers for one model
c  mmax is max number of models
