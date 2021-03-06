      SUBROUTINE HYPREF
C--SELECTS THE PREFERRED MAGNITUDE AMONG THOSE CALCULATED
      INCLUDE 'common.inc'

C--ZERO NUMBERS IN CASE NO MAG IS SELECTED, SET PMAG TO DEFAULT VALUE
      PMAG=VNOMAG
      NPMAG=0
      LABPR=' '
      PMMAD=0.

C--GO THROUGH PREFERENCE ORDER UNTIL ONE IS FOUND
      DO I=1,NMAGS

        IF (MPREF(I).EQ.1) THEN
C--CHECK FMAG
          IF (FMAG.GT.VNOMAG .AND. NFMAG.GE.MNPREF(I) .AND.
     2    FMAG.GE.AMPREF(I) .AND. FMAG.LE.AXPREF(I)) THEN
            PMAG=FMAG
            NPMAG=NFMAG
            LABPR=LABF1
            PMMAD=FMMAD
            RETURN
          END IF

        ELSE IF (MPREF(I).EQ.2) THEN
C--CHECK XMAG
          IF (XMAG.GT.VNOMAG .AND. NXMAG.GE.MNPREF(I) .AND.
     2    XMAG.GE.AMPREF(I) .AND. XMAG.LE.AXPREF(I)) THEN
            PMAG=XMAG
            NPMAG=NXMAG
            LABPR=LABX1
            PMMAD=XMMAD
            RETURN
          END IF

        ELSE IF (MPREF(I).EQ.3) THEN
C--CHECK EXTERNAL BMAG
          IF (BMAG.GT.VNOMAG .AND. NBMAG.GE.MNPREF(I) .AND.
     2    BMAG.GE.AMPREF(I) .AND. BMAG.LE.AXPREF(I)) THEN
            PMAG=BMAG
            NPMAG=NBMAG
            LABPR=BMTYP
            PMMAD=0.
            RETURN
          END IF

        ELSE IF (MPREF(I).EQ.4) THEN
C--CHECK XMAG2
          IF (XMAG2.GT.VNOMAG .AND. NXMAG2.GE.MNPREF(I) .AND.
     2    XMAG2.GE.AMPREF(I) .AND. XMAG2.LE.AXPREF(I)) THEN
            PMAG=XMAG2
            NPMAG=NXMAG2
            LABPR=LABX2
            PMMAD=XMMAD2
            RETURN
          END IF

        ELSE IF (MPREF(I).EQ.5) THEN
C--CHECK FMAG
          IF (FMAG2.GT.VNOMAG .AND. NFMAG2.GE.MNPREF(I) .AND.
     2    FMAG2.GE.AMPREF(I) .AND. FMAG2.LE.AXPREF(I)) THEN
            PMAG=FMAG2
            NPMAG=NFMAG2
            LABPR=LABF2
            PMMAD=FMMAD2
            RETURN
          END IF

C        ELSE IF (MPREF(I).EQ.6) THEN
C--CHECK PRIMARY P AMP MAG
C          IF (PAMAG.GT.0. .AND. NINT(PMUSED).GE.MNPREF(I) .AND.
C     2    PAMAG.GE.AMPREF(I) .AND. PAMAG.LE.AXPREF(I)) THEN
C            PMAG=PAMAG
C            NPMAG=PMUSED
C            LABPR=LABP1
C            PMMAD=PAMAD
C            RETURN
C          END IF

C        ELSE IF (MPREF(I).EQ.7) THEN
C--CHECK SECONDARY P AMP MAG
C          IF (PAMAG2.GT.0. .AND. NINT(PMUSD2).GE.MNPREF(I) .AND.
C     2    PAMAG2.GE.AMPREF(I) .AND. PAMAG2.LE.AXPREF(I)) THEN
C            PMAG=PAMAG2
C            NPMAG=PMUSD2
C            LABPR=LABP2
C            PMMAD=PAMAD2
C            RETURN
C          END IF

        END IF
      END DO
      RETURN
      END
