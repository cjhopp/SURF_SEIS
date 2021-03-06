      SUBROUTINE HYLINV
C--GIVEN DEPTH & DISTANCE, THIS ROUTINE CALCULATES TRAVEL TIME, ITS
C--DERIVATIVES AND EMERGENCE ANGLES AT THE SOURCE FOR ALL ARRIVALS.
C  USES THE LINVOL LINEAR GRADIENT OVER HALFSPACE CALCULATOR FROM HYPOELLIPSE
C--MODTYP IS 2
      INCLUDE 'common.inc'
      LOGICAL ALTMOD, SALMOD

C--THE FOLLOWING ARE PASSED THRU THE ARRAY A:
C DTDR      !TT DERIV WRT DISTANCE
C DTDZ      !TT DERIV WRT DEPTH
C T      !TRAVEL TIME
C AIN      !ANGLE OF EMERGENCE AT SOURCE

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

C--STATION DISTANCE
      DX=DIS(KI)
C--STZSV IS THE STATION DEPTH IN KM BELOW SEA LEVEL
      STZSV=0.
      IF (LELEV(MD)) STZSV= -0.001*JELEV(J)

C--CALCULATE TRAVEL TIME & DERIVATIVES
C--VST=VELOCITY AT STATION, VEQ=VEL AT EQ (NOT USED)
      CALL LINVOL (DX,Z1,STZSV,VGRAD(MD),VSEA(MD),THICK(MD),VHALF(MD),
     2 T,AIN,DTDR,DTDZ)

C--END OF STATION LOOP
      A(I,1)=AIN
      A(I,2)=T
      A(I,3)=DTDR
280   A(I,4)=DTDZ

      RETURN
      END
