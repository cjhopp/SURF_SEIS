      SUBROUTINE HYLST
C--DOES FINAL OUTPUT OF EVENT DATA BY STATION FOR HYPOINVERSE
      INCLUDE 'common.inc'

      LOGICAL LSHAD,LALT,LKILL,LMUSE,LSMOD
      DOUBLE PRECISION DTEMP
      CHARACTER XTEMP*5,FTEMP*5, CWT*1,C1*1,C3*3, CALCH*5,FCHAR*22
      CHARACTER XCHAR*17,XCHAR2*22,XCHAR7*7,XCHAR20*20,CA1*1,CWF*1
      CHARACTER CRES*6,STAR*1, XLABL(4)*3, STA*5, SLOC*2, MLABL(5)*3
      CHARACTER LINE*145,PRVSTA*5,LABF*1,LABX*1,LABPA*1
      CHARACTER COMP1*1,COMP3*3, PCOMP1*1,PCOMP3*3,PLOC*2
      CHARACTER SNET*2, PNET*2, CT1*1, COMPA*3, CFMP*4
      CHARACTER PTEMP*5,PCHAR*20, PACHAR*37, PACHR2*37
      CHARACTER FMNOT*1, XMNOT*1	!WEIGHTOUT CHARACTER FOR FMAG,XMAG
      CHARACTER XC3*3, FC3*3, XC2*2, FC2*2, CMAD*3, XC16*16, FC16*16
      SAVE XLABL,MLABL
      DATA XLABL /'OT ','LAT','LON','Z  '/
      DATA MLABL /'CRT','CRH','CRV','CRL','CRE'/

      PRVSTA=' '
      PCOMP1=' '
      PCOMP3=' '
      PLOC=' '
      LKILL=.FALSE.
      LALT=MODALT(MOD).GT.0
      LSMOD=MODSAL(MOD).GT.0
      LSHAD=JCA.EQ.3
      CMAD='RMS'
      IF (LMED) CMAD='MAD'

C--OUTPUT EIGENDATA AND ERROR ELLIPSE
      IF (KPRINT.GT.2 .AND. LPRT) THEN
C--OUTPUT HEADING & EIGENVALUES
        WRITE (15,1100) EIGVAL
1100    FORMAT (/14X,'EIGENVALUES'/ 5X,'(',F6.3,3F7.3,')' /6X,
     2  'EIGENVECTORS OF ADJUSTMENT', 16X,'COVARIANCE', 13X,'ERRORS')
C--COMPUTE STANDARD DEVIATIONS
        DO 150 I=1,4
          TEMP=SQRT(COVAR(I,I))

C--OUTPUT EIGENVECTORS, COVARIANCE, STD. DEV. & ERROR ELLIPSE
          WRITE (15,1101) XLABL(I),(V(I,J),J=1,4),(COVAR(I,J),J=1,4),
     2    TEMP
1101      FORMAT (1X,A3,' (',F6.3,3F7.3,') (',4F8.3,')',F8.3)
150     CONTINUE
      END IF

C--PRINT ERROR ELLIPSE AXES
      IF (KPRINT.GT.0 .AND. LPRT) WRITE (15,1102)
     2 (SERR(I),IAZ(I),IDIP(I),I=1,3)
1102  FORMAT (' ERROR ELLIPSE: <SERR AZ DIP>',
     2 3('-<',F7.2,I4,I3,'>'))

1001  FORMAT (F6.2)
1002  FORMAT (F4.2)
1003  FORMAT (F4.1)
1004  FORMAT (I4)
1009  FORMAT (F5.2)
1010  FORMAT (I1)

C--ADJUST ORIGIN IF EVENT HAS ITERATED INTO ANOTHER MINUTE.
C  DO NOT ADJUST TIME IF EVENT IS STILL BEING LOCATED.
C  NSHIF IS THE SHIFT IN .01 SEC TO SUBTRACT FROM P & S TIMES
      NSHIF=0
      IF (DONE) THEN
C--ROUND ORIGIN TIME TO NEAREST .01 SEC. THIS SHOULD PREVENT ROUND OFF ERROR
C  IN RESIDUAL CALCULATION.
        T1=.01*NINT(T1*100.)

C--AT FIRST, NT IS THE TIME SHIFT IN MINUTES, THEN IN HOURS.
        NT=(T1+600.)/60.
        NT=NT-10
        IF (ABS(NT).GT.4) THEN
          WRITE (6,1035) KYEAR2,KMONTH,KDAY,KHOUR,KMIN
          IF (LPRT) WRITE (15,1035) KYEAR2,KMONTH,KDAY,KHOUR,KMIN
1035      FORMAT (' *** ORIGIN TIME SHIFT LIMITED TO 4 MIN. IN EVENT',
     2    I4,4I3)
          NT=ISIGN(4,NT)
        END IF

C--ADJUST ORIGIN MINUTES
        T1=T1-NT*60.
        KMIN=KMIN+NT
        NSHIF=NT*6000
C--ADJUST ORIGIN HOURS
        NT=(KMIN+60.)/60.
        NT=NT-1
        KMIN=KMIN-NT*60
        KHOUR=KHOUR+NT

C--ADJUST DATE IF EVENT HAS ITERATED INTO ANOTHER DAY
        IF (KHOUR.LT.0) THEN
          KHOUR=KHOUR+24
          DTEMP=DAYJL(KYEAR2,KMONTH,KDAY)-1.+.01
          CALL JDATE (DTEMP,KYEAR2,KMONTH,KDAY,JH,JN)
        END IF
        IF (KHOUR.GT.23) THEN
          KHOUR=KHOUR-24
          DTEMP=DAYJL(KYEAR2,KMONTH,KDAY)+1.+.01
          CALL JDATE (DTEMP,KYEAR2,KMONTH,KDAY,JH,JN)
        END IF
      END IF

C--COUNT NUMBER OF VALID READINGS WITH WEIGHT CODES <4
      NVR=0
C--TALLY FIRST MOTIONS WHEN THEY ARE NON-BLANK AND P TIMES ARE WEIGHTED
      NFRM=0
      DO K=1,KSTA
        LSWT=KWT(K)/10
        LPWT=KWT(K)-10*LSWT
          IF (LPWT.LT.4 .AND. KPRK(K).NE.'  ') THEN
            NVR=NVR+1
            IF (KPRK(K)(3:3).NE.' ') NFRM=NFRM+1
          END IF
          IF (LSWT.LT.4 .AND. KSRK(K).NE.'  ') NVR=NVR+1
      END DO

C--COMPUTE LARGEST GAP AS MAXGAP
      MAXGAP=0
C--CALC DISTANCE TO NEAREST STA & NUMBER OF FIRST MOTIONS
      DMIN=32700.

C--EXAMINE ALL POSSIBLE GAPS FOR THE LARGEST BETWEEN ADJACENT STATIONS
      DO 30 IM=1,M
C--DONT COUNT UNWEIGHTED STATIONS IN GAP OR MINDIS
      IF (W(IM).LT..1) GO TO 30
C--REMOVE S FLAG FROM STATION INDEX
      K=IND(IM)
      IF (K.GT.10000) K=K-10000
C--FIND LEAST STATION DISTANCE
      IF (DIS(K).LT.DMIN) DMIN=DIS(K)
      MINGAP=360
      LAZ1=KAZEM(K)/180

C--INNER STATION LOOP
      DO 20 I=1,M
        IF (I.EQ.IM .OR. W(I).LT..1) GOTO 20
        K2=IND(I)
        IF (K2.GT.10000) K2=K2-10000
        LAZ2=KAZEM(K2)/180
        JGAP=LAZ1-LAZ2
        IF (JGAP.LE.0) JGAP=JGAP+360
        IF (JGAP.LT.MINGAP) MINGAP=JGAP
20    CONTINUE

      IF (MINGAP.GT.MAXGAP) MAXGAP=MINGAP
30    CONTINUE

C--GET REAL MAGNITUDE WEIGHTS
C      FWT=MFMAG*.01
C      XWT=MXMAG*.01

C++++++++++++++ OUTPUT THE LOCATION +++++++++++++++++
C--OUTPUT TO ARCHIVE FILE, USING ARCHIVE FILE UNIT NUMBER
      IF (LARC .AND. NWR.GE.MINSTA .AND. DONE) THEN
        CALL HYSUM (7)
C--WRITE OPTIONAL SHADOWS TO SUMMARY HEADER
        IF (LSHAD) THEN
          IF (LSHA1(1).EQ.0) THEN
            SHAD1(1)='$'
            LSHA1(1)=1
          END IF

C          IF (LPMAG) THEN
C--ADD SOME PMAG DATA TO FIRST SUMMARY SHADOW CARD ONLY IF PMAGS ARE COMPUTED
C            KAZ=PNORMN*1000.+.5
C            KEM=PNRMN2*1000.+.5
C            WRITE (7,'(A,2I4)') SHAD1(1)(1:80),KAZ,KEM
C          ELSE
C--WRITE THE SUMMARY SHADOW RECORD
            WRITE (7,'(A)') SHAD1(1)(1:LSHA1(1))
C          END IF

          DO I=2,NSHA1
            IF (LSHA1(I).GT.0) THEN
              WRITE (7,1008) SHAD1(I)(1:LSHA1(I))
            ELSE
              WRITE (7,'(''$'')')
            END IF
          END DO
        END IF
      END IF

C--DO EVENT LEVEL OPERATIONS FOR OPTIONAL MAG DATA FILE
      IF (LMAG .AND. NWR.GE.MINSTA .AND. DONE) THEN
C--WRITE SUMMARY LINE TO MAG DATA FILE
        CALL HYSUM (16)
        KQ=NINT(T1*100.)

C--PREPARE EVENT FMAG INFO
        LFMAG=NINT(FMAG*100.)
        KFMMAD=NINT(FMMAD*100.)
        LFMAG2=NINT(FMAG2*100.)
        KFMMAD2=NINT(FMMAD2*100.)

C--PREPARE XMAG INFO
        LXMAG=NINT(XMAG*100.)
        KXMMAD=NINT(XMMAD*100.)
        LXMAG2=NINT(XMAG2*100.)
        KXMMAD2=NINT(XMMAD2*100.)

C--PREPARE PREFERRED MAG INFO
        LPRMAG=NINT(PMAG*100.)
        KPRMAD=NINT(PMMAD*100.)

C--PREPARE EXTERNAL MAG INFO
        LBMAG=NINT(BMAG*100.)
      END IF

C--OUTPUT TO PRINTER
      IF (LPRT) THEN

C--STORE & OUTPUT MAGNITUDES AS A STRING WHICH IS BLANK IF NONE PRESENT
        FTEMP=' '
        XTEMP=' '
        FC16=' '
        XC16=' '
        PTEMP=' '
        IF (NFMAG.GT.0) THEN
          WRITE (FTEMP,1009) FMAG
          WRITE (FC16,1019) NFMAG,FMMAD,LABF1
1019      FORMAT (I7,F6.2,1X,A1,1X)
        END IF
        IF (NXMAG.GT.0) THEN
          WRITE (XTEMP,1009) XMAG
          WRITE (XC16,1019) NXMAG,XMMAD,LABX1
        END IF
        IF (NPMAG.GT.0 .OR. PMAG.GT.VNOMAG) WRITE (PTEMP,1009) PMAG

C--FLAG THE MODEL CODE IF AN ALTERNATE OR S MODEL WAS USED FOR SOME STATIONS
        CTEMP=' '
        IF (LALT) CTEMP='*'
        IF (LSMOD) CTEMP='S'

C--OUTPUT LOCATION TO PRINT FILE
        WRITE (15,1027) CZFLAG
1027    FORMAT(1X,21('----')/' YEAR MO DA  --ORIGIN--  --',
     2 'LAT N-  --LON W--  DEPTH-',A1,' RMS   ERH   ERZ  XMAG1 FMAG1 ',
     3 'PMAG GEOID-DEP')

        WRITE (15,1028) KYEAR2,KMONTH,KDAY, KHOUR,KMIN,T1, LAT,IS,XLTM,
     2  LON,IE,XLNM,ZREP, RMS,ERH,ERZ, XTEMP,FTEMP,PTEMP, LABPR,ZGEOID
1028    FORMAT (1X,I4,2('-',I2.2), 2X,2I2.2,F6.2, I4,A1,F5.2,
     2  I5,A1,F5.2,F7.2, 3F6.2, 3(1X,A5), A1,F8.2)

        WRITE (15,1029) CMAD,CMAD
1029    FORMAT(90X,'SOURCE',/,' NSTA NPHS  DMIN MODEL GAP ITR NFM NWR ',
     2  'NWS NVR REMRKS-AVH  N.XMG-XM',A3,'-T   N.FMG-FM',A3,
     3  '-T  L F X')

        WRITE (15,1030) KSTA,M,DMIN,CRODE(MOD),CTEMP,
     2  MAXGAP,ITR,NFRM,NWR,NWS,NVR, REMK,RMK1,RMK2, CP1,CP2,CP3,
     3  XC16,FC16, SOUCOD,FMSOU,XMSOU

1030    FORMAT (1X,I4,I5,F6.1,2X,A3,A1,
     2  6I4,1X, A3,1X,2A1, 1X,3A1,
     3  2A16, 3(1X,A1))

C--WRITE SECOND & PREF MAG INFO, & DEPTH DATUM
C        IF (NFMAG2.GT.0 .OR. NXMAG2.GT.0 .OR. NPMAG.GT.0
C     2    .OR. PMAG.GT.VNOMAG) THEN
          XCHAR=' '
          XCHAR20=' '
          XCHAR2=' '
          FCHAR=' '
          PCHAR=' '
          WRITE (15,1040) CMAD,CMAD,CMAD
1040      FORMAT (/' XMAG2-N.XMG2-XM',A3,'-T-S  FMAG2-N.FMG2-FM',A3,
     2    '-T-S  PREF.MAG-N.PMAG-PR',A3,'-T DEPDAT NS ZT TYP')

C--WRITE SECOND AMP & DUR MAGS
          IF (NXMAG2.GT.0) THEN
            WRITE (XCHAR2,1041) XMAG2,NXMAG2,XMMAD2,LABX2,XMSOU2
1041        FORMAT (F5.2,I7,F6.2,2(1X,A1))
          END IF

          IF (NFMAG2.GT.0) THEN
            WRITE (FCHAR,1041) FMAG2,NFMAG2,FMMAD2,LABF2,FMSOU2
          END IF

C--WRITE THE PREFERRED MAGNITUDE
          IF (PMAG.GT.VNOMAG) THEN
            WRITE (PCHAR,'(F5.2,I7,F6.2,1X,A1)') PMAG,NPMAG,PMMAD,LABPR
          END IF

C--WRITE THE ENTIRE DATA LINE INCLUDING DEPTH DATUM
          WRITE (15,1042) XCHAR2,FCHAR,PCHAR, IDEPDAT,JDSTA,CZFLAG,
     2    MLABL(MODTYP(MODS(1))+1)
1042      FORMAT (1X,A22,2X,A22,5X,A20, I7,I3,2X,A1,1X, A3)
C        END IF

C--WRITE THE P AMPLITUDE MAGNITUDES IF EITHER IS PRESENT
C        IF ((PMUSED.GT.0. .OR. PMUSD2.GT.0.) .AND. LPMAG) THEN
C          PACHAR=' '
C          PACHR2=' '
C          WRITE (15,1045)
C1045      FORMAT (/' PAMAG-C-N.USED-N.CLIP-PMAD-NORM-T-S  ',
C     2    'PMAG2-C-N.USED-N.CLIP-PMAD-NORM-T-S')
C
C--WRITE PRIMARY P-AMPLITUDE MAGNITUDE
C          IF (PMUSED.GT.0.) THEN
C            WRITE (PACHAR,1048) PAMAG,PMUSED,PMCLIP,PAMAD,
C     2      PNORMN,LABP1,PSOUR
C1048        FORMAT (F5.2, 2X,F6.1,F7.1, F6.2, F5.2, 1X,A1,1X,A1,1X)
C          END IF
C
C--WRITE SECONDARY P-AMPLITUDE MAGNITUDE
C          IF (PMUSD2.GT.0.) THEN
C            WRITE (PACHR2,1048) PAMAG2,PMUSD2,PMCLP2,PAMAD2,
C     2      PNRMN2,LABP2,PSOUR2
C          END IF
C
C--FLAG THE PAMAGS WITH A + IF ANY OF THE 3 MIN NORM STATIONS ARE CLIPPED
C  (PAMAG IS A MIN. MAG)
C--THIS MINIMUM FLAG IS ALSO SET FROM THE RATIO OF PMCLIP TO PMUSED
C          IF (MINPM.EQ.1) PACHAR(7:7)='+'
C          IF (MINPM2.EQ.1) PACHR2(7:7)='+'
C
C--WRITE THE ENTIRE DATA LINE
C          WRITE (15,'(1X,2A37)') PACHAR,PACHR2
C        END IF
C
C--WRITE EXTERNAL (BERKELEY) MAGNITUDE
C          IF (BMAG.GT.0. .OR. MBMAG.GT.0) THEN
C          XWT=MBMAG*.01
C          WRITE (15,1043) BMAG,XWT,BMTYP
C1043      FORMAT (/' BERKELEY MAG=',F5.2,'  NUMBER READINGS=',F5.1,

        IF (BMAG.GT.VNOMAG .OR. NBMAG.GT.0) THEN
          WRITE (15,1043) BMAG,NBMAG,BMTYP
1043      FORMAT (/' EXTERNAL MAG=',F5.2,'  NUMBER READINGS=',I5,
     2    '  TYPE=',A1)
        END IF

C--WRITE EXTERNAL (AMP) X-MAGNITUDE
C--ASSUME FOR NOW THAT A ZERO OR BLANK EXTERNAL MAG MEANS NO MAG
C        IF (BMAGX.GT.VNOMAG .OR. NBMAGX.GT.0) THEN
        IF (BMAGX.GT.0. .OR. NBMAGX.GT.0) THEN
          WRITE (15,1044) BMAGX,NBMAGX,BMTYPX
1044      FORMAT (' EXTERNAL XMAG=',F5.2,'  NUMBER READINGS=',I5,
     2    '  TYPE=',A1)
        END IF

C--WRITE REGION NAME & MULTPLE CRUSTAL MODELS USED
        IF (LMULT) THEN
          WRITE(15,1014) FULNAM,
     2    (CRODE(MODS(I)), WMOD(I),I=1,NMOD)
1014      FORMAT (' REGION= ',A25,:
     2    '  MODELS USED:',3(2X,A3,'=',F4.2))
        ELSE
          WRITE (15,1014) FULNAM
        END IF
      END IF

C+++++++++++++++ PRINT OUT THE STATION LIST +++++++++++++++++++++++

      LINCUT=0
      MOUT=0
C--PRINT THE STATION HEADING
      IF (KPRINT.GT.0 .AND. LPRT) THEN
        WRITE (LINE,1031)
1031    FORMAT(' STA NET COM L CR DIST AZM  AN P/S WT   SEC (TOBS ',
     2  '-TCAL -DLY  =RES)  WT   SR  INFO  CAL  DUR-W-FMAG-T',
     3  ' -AMP-U-PER-W-XMAG-T DEV')
        WRITE (15,'(/,A)') LINE(1:124)
      END IF

C--LIST STATIONS IN DISTANCE ORDER
C--DISLST IS THE LEAST REMAINING DISTANCE AND WILL BE PRINTED ON THIS PASS
C--DISNXT IS THE NEXT TO LEAST REMAINING DISTANCE,
C--AND WILL BE PRINTED ON THE NEXT PASS
      DISNXT=-1

C--START OUTER STATION LOOP
40    DISLST=DISNXT
      DISNXT=32700
      KLSTA=-1

C--START INNER STATION LOOP
      DO 90 IM=1,M

C--DETERMINE STATION INDEX & WHETHER IT IS P OR S
      K=IND(IM)
      KPS=K/10000
      K=K-10000*KPS
      DK1=DIS(K)

      IF (DK1.NE.DISLST) THEN
        IF (DK1.GT.DISLST .AND. DK1.LT.DISNXT) DISNXT=DK1
        GOTO 90
      END IF
      MOUT=MOUT+1

C--DECODE CALCULATED TRAVEL TIME
      TCAL=.01*MTCAL(IM)

C--WRITE A LINE OF PHASE ARRIVAL DATA FOR THIS STATION
C--PREPARE STATION INFO
      IF (K.EQ.KLSTA) GOTO 60
      J=KINDX(K)

C--DECODE DISTANCE, AZIMUTH AND EMERGENCE ANGLE
      KAZ=KAZEM(K)/180
      KEM=ABS(KAZEM(K)-180*KAZ)
      IF (KAZ.LT.0) KAZ=KAZ+360

C--NOW STORE XMAG AND FMAG AND WEIGHTS AS ALPHAMERIC CODE
      CALCH=' '
      FCHAR=' '
      XCHAR=' '
      XCHAR20=' '
      LABF=' '
      LABX=' '

      IF (FMPK(K).GT.0. .OR. AMPK(K).GT.0.) THEN
C--USE KCAL INSTEAD OF JCAL IF KCAL IS PRESENT
        IF (KCAL(K).EQ.0) THEN
          TEMP=JCAL(J)*.001
        ELSE
          TEMP=KCAL(K)*.01
        END IF
        WRITE (CALCH,1009) TEMP
      END IF

      FMNOT=' '
      CFMP='    '
      IF (FMPK(K).GT.0.) THEN
C--FORMAT DURATION TO FIT IN 4 COLS
        IF (FMPK(K).LT.10.) THEN
          WRITE (CFMP,'(F4.2)') FMPK(K)
        ELSE IF (FMPK(K).LT.100.) THEN
          WRITE (CFMP,'(F4.1)') FMPK(K)
        ELSE
          KFMP=NINT(FMPK(K))
          WRITE (CFMP,'(I4)') KFMP
        END IF

        TEMP=.01*KFMAG(K)
C--CHOOSE THE CORRECT DUR MAG TYPE CODE (USE CODE2 IF COMPONENT USED FOR BOTH)
        IF (JFM1(J)) LABF=LABF1
        IF (JFM2(J)) LABF=LABF2

        WRITE (FCHAR,'(A4,I2,F5.2,1X,A1)') CFMP,KFWT(K),TEMP,LABF
        IF (KFWT(K).EQ.0) FCHAR(6:6)=' '

C--IDENTIFY STATIONS WITH NO WEIGHT
        IF (JFWT(J).LT.2 .OR. KFWT(K).GT.3 .OR. 
     2   (.NOT.JFM1(J) .AND. .NOT.JFM2(J))) FMNOT='X'
        FCHAR(12:12)=FMNOT
C--BLANK OUT NON-MAGNITUDES
        IF (TEMP.LE.VNOMAG) FCHAR(7:13)=' '
      END IF

C--LOAD AMP MAG DATA INTO OUTPUT STRING
C--PUT A ZERO IN AMP FIELD BECAUSE READING A NUM FIELD WITH ALL BLANKS IS BAD
      XMNOT=' '
      XCHAR7='      0'
      IF (AMPK(K).GT.0.) THEN
        TEMP=.01*KXMAG(K)
        TEMP3=AMPK(K)

C--USE PERIOD FROM PHASE CARD IF IT WAS GIVEN, OTHERWISE FROM STATION CARD
        IF (KPER(K).GT.0) THEN
          TEMP2=.01*KPER(K)
        ELSE
          TEMP2=.1*JPER(J)
        END IF
        IF (TEMP2.GT.9.9) TEMP2=9.9

C--CHOOSE THE CORRECT AMP MAG TYPE CODE (USE CODE2 IF COMPONENT USED FOR BOTH)
        LABX=' '
C        IF (KIMTYP(K).EQ.1) LABX='L'
C        IF (KIMTYP(K).EQ.2) LABX='X'
        IF (JXM1(J)) LABX=LABX1
        IF (JXM2(J)) LABX=LABX2

C--LABEL THE AMPLITUDE WITH UNITS
        CA1='M'
        IF (KAMPU(K).EQ.2) CA1='C'
        IF (KAMPU(K).EQ.3) CA1='D'
        IF (KAMPU(K).EQ.4) CA1='H'

C--CHOOSE THE AMP TYPE CODE LABEL
        CT1=' '
        IF (KAMPTYP(K).EQ.1) CT1='W'
        IF (KAMPTYP(K).EQ.2) CT1='V'
        IF (KAMPTYP(K).EQ.3) CT1='A'
        IF (KAMPTYP(K).EQ.4) CT1='X'
        IF (KAMPTYP(K).EQ.5) CT1='D'

C--SET THE WEIGHTOUT CODE TO X IF THE STATION AMP MAG WAS NOT USED (WEIGHTED)
C  IN THE EVENT MAGNITUDE.  LMUSE IS TRUE IF MAG WOULD BE USED IN ONE OF XMAG1
C  OR XMAG2 ACCORDING TO COMPONENT (OR INSTRUMENT) AND TYPE SELECTION.
        LMUSE=(JXM1(J) .AND. 
     2  (MAG1TYPX.EQ.0 .OR. KIMTYP(K).EQ.MAG1TYPX)) .OR.
     3  (JXM2(J) .AND. (MAG2TYPX.EQ.0 .OR. KIMTYP(K).EQ.MAG2TYPX))
     
        XMNOT='X'
        IF (LMUSE .AND. KXWT(K).LT.4 .AND. JXWT(J).GT.0) XMNOT=' '

C--XCHAR20 IS USED FOR PRINT FILE, XCHAR FOR OLD ARCHIVE FORMAT
        IF (AMPK(K).LT.0.996) THEN
          WRITE (XCHAR20,'(F6.3,A1,F4.2,A1,I1,F5.2,2A1)')
     2    TEMP3,CA1,TEMP2,CT1,KXWT(K),TEMP,XMNOT,LABX
          WRITE (XCHAR,'(F4.2)') TEMP3

        ELSE IF (AMPK(K).LT.9.9) THEN
          WRITE (XCHAR20,'(F6.3,A1,F4.2,A1,I1,F5.2,2A1)')
     2    TEMP3,CA1,TEMP2,CT1,KXWT(K),TEMP,XMNOT,LABX
          IF (TEMP3.GT.9.9) TEMP3=9.9
          WRITE (XCHAR,'(F4.1)') TEMP3

        ELSE
          ITMP=NINT(TEMP3)
          IF (ITMP.GT.999999) ITMP=999999
          WRITE (XCHAR20,'(I6,A1,F4.2,A1,I1,F5.2,2A1)')
     2    ITMP,CA1,TEMP2,CT1,KXWT(K),TEMP,XMNOT,LABX
          IF (ITMP.GT.9999) ITMP=9999
          WRITE (XCHAR,'(I4)') ITMP
        END IF

C--MAKE OUTPUT MORE READABLE
        IF (KXWT(K).EQ.0) XCHAR20(13:13)=' '
        IF (XCHAR20(8:8).EQ.'0') XCHAR20(8:8)=' '
        IF (TEMP.LE.VNOMAG) XCHAR20(14:19)='     '

C--XCHAR7 IS USED FOR YR 2000 ARCHIVE FORMAT. FIELD IS READ F7.2
C  PLACE DECIMAL POINT TO GET MAX DYNAMIC RANGE
        IF (TEMP3.LE.0.) THEN
          XCHAR7='      0'
        ELSE IF (TEMP3.LT.9.9999 .AND. TEMP3.GE.0.) THEN
          WRITE (XCHAR7,'(F7.5)') TEMP3
        ELSE IF (TEMP3.LT.99.999 .AND. TEMP3.GE.9.9999) THEN
          WRITE (XCHAR7,'(F7.4)') TEMP3
        ELSE IF (TEMP3.LT.999.99 .AND. TEMP3.GE.99.999) THEN
          WRITE (XCHAR7,'(F7.3)') TEMP3
        ELSE IF (TEMP3.LT.9999.9 .AND. TEMP3.GE.999.99) THEN
          WRITE (XCHAR7,'(F7.2)') TEMP3
        ELSE IF (TEMP3.LT.999999. .AND. TEMP3.GE.9999.9) THEN
          WRITE (XCHAR7,'(F7.0)') TEMP3
        ELSE IF (TEMP3.GE.999999.) THEN
          XCHAR7='999999.'
        END IF

      END IF

C--DECODE ASSIGNED WEIGHTS
      LSWT=KWT(K)/10
      LPWT=KWT(K)-LSWT*10
C--GIVE THESE A VALUE FOR ARCHIVE OUTPUT IN CASE AN ARRIVAL IS NOT PRESENT
      IMPORP=0
      IMPORS=0
      KPWT=0
      KSWT=0
      KPRES=0
      KSRES=0
      KPDLY=0
      KSDLY=0

C--SET STATION P DELAY

60    PSFAC=0.
C--DLY IS THE STATION DELAY IN SEC, PSFAC AND PSB TRANSLATE P-TT TO S-TT
C--PREPARE ARRIVAL TIME INFO, BUT FIRST DECIDE IF THIS IS P OR S
      IF (KPS.EQ.0) THEN

C--ASSUME A P ARRIVAL
        PSFAC=1.
        PSB=0.
        TEMP2=KP(K)
        IF (LMULT) THEN
          DLY=0.
          DO I=1,NMOD
            IT=MODS(I)
            DLY=DLY+ (0.01*WMOD(I)*JPD(IT,J))
          END DO
        ELSE
          DLY=0.01*JPD(MOD,J)
        END IF

        KPDLY=NINT(100.*DLY)
        C3=KPRK(K)
        C1=' '
        LWT=LPWT
C        KP(K)=KP(K)-NSHIF

C .. 8/15/95 AWW I2 INTEGER OVERFLOW PROBLEM
        ITMP = KP(K) - NSHIF
        IF (ITMP .LT. -32768) THEN
          ITMP = -32768
        ELSE IF (ITMP .GT. 32767) THEN 
          ITMP = 32767
        END IF      
        KP(K) = ITMP

        SEC=KP(K)*.01
        IMPORP=IMPORT(IM)
        KPWT=100.*W(IM)+.5
        IF (KPWT.GT.999) KPWT=999
      ELSE

C--ASSUME AN S ARRIVAL
C--MULTIPLE MODEL
        IF (LMULT) THEN
          PSFAC=0.
          PSB=0.
          DLY=0.
          DO I=1,NMOD
            IT=MODS(I)

C--IF A SEPERATE S MODEL IS DEFINED, USE THAT
            IF (MODSAL(IT).GT.0) THEN
              IT=MODSAL(IT)
              PSFAC=PSFAC+ WMOD(I)
              PSB=PSB
              DLY=DLY +(WMOD(I)*0.01*JPD(IT,J))
            ELSE
              IT=MODS(I)
              PSFAC=PSFAC+ WMOD(I)*POSM(IT)
              PSB=PSB+ WMOD(I)*POSB(IT)
              DLY=DLY +(WMOD(I)*0.01*POSM(IT)*JPD(IT,J))
            END IF
          END DO

C--SINGLE MODEL
        ELSE
C--IF A SEPERATE S MODEL IS DEFINED, USE THAT AND
C  TIMES AND DELAYS DO NOT HAVE TO MULTIPLIED BY POS
          IF (MODSAL(MOD).GT.0) THEN
            IT=MODSAL(MOD)
            PSFAC=1.
            PSB=0.
            DLY=0.01*JPD(IT,J)
          ELSE
            IT=MOD
            PSFAC=POSM(IT)
            PSB=POSB(IT)
            DLY=0.01*POSM(IT)*JPD(IT,J)
          END IF
        END IF

        KSDLY=NINT(100.*DLY)
        C3=KSRK(K)
        C1='S'
        LWT=LSWT
C        KS(K)=KS(K)-NSHIF

C .. 8/15/95 AWW I2 INTEGER OVERFLOW PROBLEM
        ITMP = KS(K) - NSHIF
        IF (ITMP .LT. -32768) THEN
          ITMP = -32768
        ELSE IF (ITMP .GT. 32767) THEN 
          ITMP = 32767
        END IF      
        KS(K) = ITMP

        SEC=KS(K)*.01
        TCAL=(TCAL*PSFAC) +PSB
        IMPORS=IMPORT(IM)
        KSWT=100.*W(IM)+.5
        IF (KSWT.GT.999) KSWT=999
      END IF

C--SET PARAMETERS COMMON TO BOTH P & S
      TOBS=SEC-T1
      XIMPOR=IMPORT(IM)*.001
C--OUTPUT WEIGHT AS A STRING
      CWT=' '
      IF (LWT.GT.0) WRITE (CWT,1010) LWT

C--FLAG STATIONS WHICH USED AN ALTERNATE MODEL
      CTEMP=' '
      IF (LALT .AND. JLMOD(J)) CTEMP='*'
      IF (LSMOD .AND. KPS.EQ.1) CTEMP='S'

C--OUTPUT RESIDUAL AS A STRING AND FLAG IT IF LARGE
      STAR=' '
      RES=TOBS-TCAL-DLY
C--LIMIT THIS NUMBER FOR PRINT OUTPUT ONLY
      IF (TOBS.GT.999.99) TOBS=999.99
      CRES=' '
      IF (C3.NE.'   ') THEN
        WRITE (CRES,1001) RES
C--FLAG READING WITH A * IF LARGE OR AN X IF READING NOT USED
        IF (ABS(RES).GT..5) STAR='*'
C        IF (LWT.GT.3 .OR. KPSWT(K).NE.' ') STAR='X'
        IF (LWT.GT.3) STAR='X'
      END IF

C--SET & THEN LIMIT P & S RESIDUALS AS INTEGERS FOR OUTPUT TO ARCHIVE FILE
      IF (KPS.EQ.0) THEN
C--P WAVE
        KPRES=NINT(100.*RES)
        IF (KPRES.GT.9999) KPRES=9999
        IF (KPRES.LT.-999) KPRES=-999
      ELSE
C--S WAVE
        KSRES=NINT(100.*RES)
        IF (KSRES.GT.9999) KSRES=9999
        IF (KSRES.LT.-999) KSRES=-999
      END IF

      STA=STANAM(J)
      SNET=JNET(J)
      COMP1=JCOMP1(J)
      COMP3=JCOMP3(J)
      COMPA=JCOMPA(J)
      SLOC=JSLOC(J)
C--USE LOCATION CODE AS INPUT, DO NOT ASSUME WHOLE EVENT IS THE SAME
C      IF (LLOC2) THEN
C        SLOC=JSLOC2(J)
C      ELSE
C        SLOC=JSLOC(J)
C      END IF

C--PRINT ARRIVAL TIME INFORMATION FOR ONE STATION
C--OPTIONALLY DONT PRINT STATION IF THERE IS NO DATA FOR IT. USE WEIGHTS
C  ASSIGNED BY THE USER TO DECIDE IF VALID DATA IS THERE.
      IF (KPRINT.GT.0 .AND. LPRT .AND. (LPRALL .OR. 
     1 (LWT.LT.4) .OR.
     2 (KFWT(K).LT.4 .AND. FMPK(K).GT.0.) .OR. 
     4 (KPAMP(K).GT.0 .AND. LPPRT) .OR.
     3 (KXWT(K).LT.4 .AND. AMPK(K).GT.0.))) THEN

C--DECIDE WHETHER TO PRINT STATION NAME ON DATA LINE. PRINT IT IF THIS STATION
C  IS DIFFERENT FROM PREVIOUS LINE OR IF THE STATION IS THE SAME BUT THE
C  PREVIOUS LINE WAS NOT PRINTED BECAUSE IT HAD NO DATA.
        IF (K.NE.KLSTA .OR. LKILL) THEN

C--WRITE ALL DATA FOR FIRST PHASE (WHICH IS A P UNLESS ONLY S WAS GIVEN)
          WRITE(LINE,1032)
     1    STA,SNET,CTEMP, COMP3,SLOC,COMP1,STRMK(J), DK1,KAZ,KEM,
     2    C3,CWT, SEC,TOBS,TCAL,
     3    DLY,CRES,STAR, W(IM),C1,KSOU(K),
     4    KRMK(K),XIMPOR, CALCH,FCHAR,XCHAR20, KRMK6(K)

1032      FORMAT (1X,A5,A2,A1, A3,1X,A2,2A1, F5.1,2I4,1X,
     2    A3,1X,A1,1X, 3F6.2,
     3    F5.2,A6,A1, F5.2,A1,1X,A1,
     4    A1,F6.3, A5,1X,A13,A20,1X, A6)

C--BLANK OUT REMARKS SINCE ONLY READING PHASE CARDS RESETS REMARK FIELD
          KRMK6(K)='    '      !

C--OMIT PRINTING STATION NAME IF SAME AS PREVIOUS ONE, EX FOR COL 9
          IF (STA.EQ.PRVSTA .AND. SNET.EQ.PNET .AND. COMP3.EQ.PCOMP3
     2     .AND. SLOC.EQ.PLOC) THEN
            LINE(1:8)=' '
            LINE(10:30)=' '
          END IF
C--BLANK OUT WEIGHT FIELDS OF UNWEIGHTED STATIONS
          IF (STAR.EQ.'X') THEN
            LINE(69:72)=' '
            LINE(78:82)=' '
          END IF

C--ADD THE DIGITIZER DEVICE CODE
          LINE(122:124)=KDEV(K)

C--WRITE THE STATION LINE
C--OPTIONALLY WRITE P AMPLITUDE MAG INFO. THIS OVERWRITES THE REMARK FIELD
C          IF (LPPRT .AND. KPAMP(K).GT.0) THEN
C
C--CHOOSE THE CORRECT P AMP MAG TYPE CODE (USE 2 IF COMPONENT USED FOR BOTH)
C            IF (JPM1(J)) THEN
C              LABPA=LABP1
C              ONORM=PNORM(K)
C            END IF
C            IF (JPM2(J)) THEN
C              LABPA=LABP2
C              ONORM=PNORM2(K)
C            END IF
C
C            TEMP=KPMAG(K)*.01
C            WRITE (LINE(119:145),1051) KPAMP(K), PARMK(K), KPAWT(K),
C     2      TEMP, PAWT(K), ONORM, LABPA
C1051        FORMAT (I5,1X,A1,I1, 2F5.2,F5.3,1X,A1)
C
C--PRINT BLANK INSTEAD OF 0 FOR FULLY WEIGHTED STATIONS
C            IF (LINE(126:126).EQ.'0') LINE(125:125)=' '
C--PRINT A + NEXT TO MINIMUM MAGS FROM CLIPPED STATIONS
C            IF (KPAWT(K).GT.1) LINE(128:128)='+'
C--PRINT AN X NEXT TO UNWEIGHTED MAGS
C            IF (PAWT(K).EQ.0.) LINE(128:128)='X'
C--BLANK OUT SOME LEADING ZEROS FOR BETTER READABILITY
C            IF (LINE(129:129).EQ.'0') LINE(129:129)=' '
C            IF (LINE(133:133).EQ.'0') LINE(133:133)=' '
C
C            WRITE (15,'(A)') LINE(1:145)
C

C--SHORTER OUTPUT LINE
C          ELSE
            WRITE (15,'(A)') LINE(1:125)
C          END IF

          PRVSTA=STA
          PNET=SNET
          PCOMP3=COMP3
          PLOC=SLOC

        ELSE

C--WRITE ONLY ARRIVAL TIME INFO FOR AN S FOLLOWING A P
          WRITE (LINE,1033) C3,CWT, SEC,TOBS,TCAL,DLY,
     2    CRES,STAR,W(IM), C1,KSOU(K),KRMK(K),XIMPOR

1033      FORMAT (31X,A3,1X,A1,1X, 3F6.2,F5.2,
     2    A6,A1,F5.2, A1,1X,2A1,F6.3)
C--ALSO LOAD S INTO OUTPUT LINE BECAUSE CTEMP WILL NOT BE WRITTEN ON S AFTER P
          IF (LSMOD .AND. KPS.EQ.1) LINE(9:9)='S'

C--BLANK OUT WEIGHT FIELDS OF UNWEIGHTED STATIONS
          IF (STAR.EQ.'X') THEN
            LINE(69:72)=' '
            LINE(78:82)=' '
          END IF
          WRITE (15,'(A)') LINE(1:82)
        END IF
        LKILL=.FALSE.

C--KEEP TRACK OF NUMBER OF STATIONS NOT PRINTED
      ELSE
        LINCUT=LINCUT+1
        LKILL=.TRUE.
      END IF
C---------------------------------------------------------------
C--WRITE A RECORD CONTAINING ALL STATION INFO TO AN ARCHIVE FILE
C--DON'T WRITE A LINE YET IF S ARRIVAL IS TO COME
      IF (IM.LT.M) THEN
        I=IND(IM+1)
        K1=I/10000
        K1=I-K1*10000
        IF (K.EQ.K1) GO TO 80
      END IF
      KTEMP=NINT(DIS(K)*10.)
      IF (KTEMP.GT.9999) KTEMP=9999

C--MANAGE MAGNITUDE STRINGS FOR BOTH ARC AND MAG FILES
      IF (KFMAG(K).LT.-99) THEN
        TEMP=ANINT(KFMAG(K)*.01)
        WRITE (FC3,'(F3.0)') TEMP
      ELSE
        WRITE (FC3,'(I3)') KFMAG(K)
      END IF
      IF (KXMAG(K).LT.-99) THEN
        TEMP=ANINT(KXMAG(K)*.01)
        WRITE (XC3,'(F3.0)') TEMP
      ELSE
        WRITE (XC3,'(I3)') KXMAG(K)
      END IF
          
      IF (LARC .AND. DONE .AND. NWR.GE.MINSTA) THEN

C--THIS PREVENTS ACCIDENTAL DATA OVERFLOW IN CASE OF VERY BAD TIMES
        IF (KP(K).LE.-10000) THEN
          KP(K)=-9999
          LPWT=9
        END IF
        IF (KS(K).LE.-10000) THEN
          KS(K)=-9999
          LSWT=9
        END IF

        IF (L2000) THEN
C--YEAR 2000 FORMAT
          WRITE (7,1005) STA,SNET,COMP1,COMP3, KPRK(K),LPWT,
     2    KYEAR2,KMONTH,KDAY,KHOUR,KMIN, KP(K),KPRES,KPWT,
     3    KS(K),KSRK(K),LSWT, KSRES,XCHAR7,KAMPU(K),KSWT,
     4    KPDLY,KSDLY,KTEMP,
     5    KEM,KXWT(K),KFWT(K),KPER(K), KRMK(K),CFMP,KAZ,
     6    FC3,XC3, IMPORP,IMPORS, KSOU(K),LABF,LABX,
     7    SLOC,KAMPTYP(K), COMPA,XMNOT,FMNOT

1005      FORMAT (A5,A2,1X,A1,A3, 1X,A3,I1,
     2    I4,4I2.2, I5,I4,I3,
     3    I5,A2,1X,I1, I4,A7,I2,I3,
     4    3I4,
     5    I3,2I1,I3, A1,A4,I3,
     6    2A3, 2I4, 3A1,
     7    A2,I2, A3,2A1)

        ELSE
C--OLD FULL FORMAT
          IXTMP=NINT(.1*KXMAG(K))
          IFTMP=NINT(.1*KFMAG(K))
          IF (IXTMP.LT.-9) IXTMP=-9
          IF (IFTMP.LT.-9) IFTMP=-9
C--PUT A ZERO IN AMP FIELD BECAUSE READING A NUM FIELD WITH ALL BLANKS IS BAD
          IF (XCHAR(2:4).EQ.'   ') XCHAR(2:4)='  0'

          WRITE (7,1034) STA(1:4),KPRK(K),LPWT,COMP1,
     2    KYEAR,KMONTH,KDAY,KHOUR,KMIN, KP(K),KPRES,KPWT,
     3    KS(K),KSRK(K),LSWT, KSRES,XCHAR(2:4),KSWT, KPDLY,KSDLY,KTEMP,
     4    KEM,KXWT(K),KFWT(K),KPER(K), KRMK(K),CFMP,KAZ,
     5    IFTMP,IXTMP, IMPORP,IMPORS, KSOU(K),LABF,LABX,
     6    STA(5:5),COMP3,SNET,SLOC

1034      FORMAT (A4,A3,I1,A1,
     2    5I2.2, I5,I4,I3,
     3    I5,A2,1X,I1, I4,A3,I3, 3I4,
     4    I3,2I1,I3, A1,A4,I3,
     5    2I2, 2I4, 1X,3A1,
     6    A1,A3,2A2)
        END IF

C--WRITE OPTIONAL SHADOW RECORD
        IF (LSHAD) THEN
          IF (KLSHA(K).EQ.0) THEN
            WRITE (7,'(''$'')') 
          ELSE

C--ADD P AMPLITUDE MAGNITUDE CALCULATIONS TO SHADOW CARD IF PMAG CALCULATED 
C            IF (LPMAG .AND. KPAMP(K).GT.0) THEN
C              IFTMP=0
C              IF (JPM1(J)) IFTMP= NINT(PNORM(K)*1000.)
C              IF (JPM2(J)) IFTMP= NINT(PNORM2(K)*1000.)
C              IXTMP= NINT(PAWT(K)*100.)
C
C--CHOOSE THE CORRECT P AMP MAG TYPE CODE (USE 2 IF COMPONENT USED FOR BOTH)
C              IF (JPM1(J)) LABPA=LABP1
C              IF (JPM2(J)) LABPA=LABP2

C              WRITE (KSHAD(K)(93:103),1059) KPMAG(K),IXTMP,IFTMP,LABPA
C1059          FORMAT (2I3,I4,A1)
C              KLSHA(K)=103
C            END IF

            WRITE (7,1008) KSHAD(K)(1:KLSHA(K))
1008        FORMAT (A)
          END IF
        END IF
      END IF

C------------ WRITE STATION TO MAGNITUDE DATA FILE --------------------
C--WRITE ENTIRE LINE, EVEN IF SOME MAGNITUDES ARE NOT PRESENT
      IF (LMAG .AND. DONE .AND. NWR.GE.MINSTA .AND.
     2 (AMPK(K).GT.0 .OR. FMPK(K).GT.0.)) THEN

C--PREPARE STATION MAG WEIGHTS AS CHARACTERS
        IF (JFWT(J).EQ.10) THEN
          CWF=' '
        ELSE
          CWF=CHAR(48+JFWT(J))
        END IF
        IF (JXWT(J).EQ.10) THEN
          CWT=' '
        ELSE
          CWT=CHAR(48+JXWT(J))
        END IF

C--DETERMINE DUR MAG COMPONENT CORRECTION
        ICOMF=0
        DO I=1,NFCM
          IF (JCOMP3(J)(1:NCOMP) .EQ. CFCM(I)(1:NCOMP))
     2    ICOMF=NINT(100.*AFCM(I))
        END DO

C--DETERMINE AMP MAG COMPONENT CORRECTION
        ICOMX=0
        DO I=1,NXCM
          IF (JCOMP3(J)(1:NCOMP) .EQ. CXCM(I)(1:NCOMP)) 
     2    ICOMX=NINT(100.*AXCM(I))
        END DO

C--WRITE STATION LINE
        WRITE (16,1036) STA,SNET,COMP3,SLOC,
     1  KYEAR2,KMONTH,KDAY,KHOUR,KMIN, KQ,KTEMP,
     2  KPRK(K),LPWT,MTCAL(IM),KPRES, REMK,KSOU(K),KRMK(K),
     3  JTYPE(J),JCAL(J), 
     4  CFMP,KFWT(K), CWF,JFCOR(J),ICOMF, KFMAG(K),LABF,
     5  AMPK(K),CA1,KPER(K),KXWT(K),CWT,JXCOR(J),ICOMX,KXMAG(K),LABX, 

     6  LFMAG,LABF1,FMSOU,KFMMAD,NFMAG,
     7  LFMAG2,LABF2,FMSOU2,KFMMAD2,NFMAG2,
     8  LXMAG,LABX1,XMSOU,KXMMAD,NXMAG,
     9  LXMAG2,LABX2,XMSOU2,KXMMAD2,NXMAG2,

     1  LPRMAG,LABPR,KPRMAD,NPMAG,
     2  LBMAG,BMTYP,NBMAG

1036    FORMAT (A5,A2,A3,A2,1X,
     1  I4,4I2.2, 2I4,
     2  A2,I1,2I4, A3,2A1,
     3  I1,I5,3X, 
     4  A4,I1, A1,2I3,I4,A1,2X,
     5  F6.2,A1,I3,I1, A1,2I3,I4,A1,2X,

     6  4(I4,2A1,I3,I4,1X),
     1  I4,A1,1X,I3,I4,1X,
     2  I4,A1,I3)

      END IF

C--UPDATE THE LAST STATION INDEX; END OF BOTH STATION LOOPS
80    KLSTA=K
90    CONTINUE
      IF (MOUT.LT.M) GOTO 40

C--WRITE NUMBER OF STATIONS NOT PRINTED
      IF (KPRINT.GT.0 .AND. LPRT .AND. LINCUT.GT.0) WRITE (15,
     2 '(I5,'' UNWEIGHTED STATIONS NOT PRINTED.'')') LINCUT

C--OUTPUT TERMINATOR LINE TO ARCHIVE FILE IF NO MORE PHASE CARDS REMAIN
      IF (DONE .AND. NWR.GE.MINSTA .AND. .NOT.LTBIG) THEN

C--FIRST WRITE UNKNOWN STATIONS SAVED BY HYPHS, IF ANY
        IF (LARC) THEN
          DO I=1,NUNK
            WRITE (7,1008) PUNK(I)
            IF (LSHAD) WRITE (7,1008) SUNK(I)(1:NSUNK(I))
          END DO
        END IF

C--WRITE TERMINATOR CARD COPIED FROM INPUT
        LTERM=LENG(TERM)
        IF (LARC) WRITE (7,'(A)') TERM(1:LTERM)
        IF (LMAG) WRITE (16,'(A)') TERM(1:LTERM)

C--WRITE OPTIONAL SHADOW RECORD
        IF (LARC .AND. LSHAD) THEN
          IF (LENSHA.GT.0) THEN
            WRITE (7,1008) SHADO(1:LENSHA)
          ELSE
            WRITE (7,'(''$'')')
          END IF
        END IF
      END IF

      RETURN
      END
