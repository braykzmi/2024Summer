PARENT_DIR="/rsrch1/ip/bmi/trainingdata"

TRAINING_DIR="$PARENT_DIR/raw_trainingdata"

CSV_FILE="$PARENT_DIR/mrn_casenum.csv"

OUTPUT_DIR="$PARENT_DIR/tmp"

CASE_DIR="$PARENT_DIR/cases"

LABEL1_DIR="$PARENT_DIR/biliary_labels"

LABEL2_DIR="$PARENT_DIR/liver_labels"

LABEL3_DIR="$PARENT_DIR/tumor_labels"

SEGMENTATIONS_DIR="$PARENT_DIR/segmentations"

CASE_NUM=0

echo "MRN,CASE_NUM">$CSV_FILE

pad_casenum(){
printf "%03d" $1
}

find $TRAINING_DIR -mindepth 2 -maxdepth 2 -type d | while read STUDY_DIR; do
MEDRECNUM=$(basename $(dirname $STUDY_DIR))

CASE_NAME="case_$(pad_casenum $CASE_NUM)"
CASE_CHANNEL="case_$(pad_casenum $CASE_NUM)_0000"


RTSTRUCT_DIR=""
DICOM_DIR=""

for DIR in "$STUDY_DIR"/*; do
if [ -d "$DIR" ]; then
FILE_COUNT=$(find "$DIR" -type f | wc -l)
if [ "$FILE_COUNT" -eq 1 ]; then
RTSTRUCT_DIR="$DIR"
else
DICOM_DIR="$DIR"
fi
fi
done

RTSTRUCT_FILE=$(find "$RTSTRUCT_DIR" -type f)

dcmrtstruct2nii convert -r "$RTSTRUCT_FILE" -d "$DICOM_DIR" -o "$OUTPUT_DIR"

T=""
B="$OUTPUT_DIR/mask_Biliary.nii.gz"
L="$OUTPUT_DIR/mask_Liver.nii.gz"

c3d "$B" -thresh 0 inf 1 1 -o "$OUTPUT_DIR/ones.nii.gz"
O="$OUTPUT_DIR/ones.nii.gz"

if [ -f "$OUTPUT_DIR/mask_Tumor.nii.gz" ]; then
T="$OUTPUT_DIR/mask_Tumor.nii.gz"
c3d "$T" -thresh 1 inf 1 0 -o "$T"
else
c3d "$B" -thresh 1 inf 0 0 -o "$OUTPUT_DIR/zeros.nii.gz"
T="$OUTPUT_DIR/zeros.nii.gz"
fi

c3d "$B" -thresh 1 inf 1 0 -o "$B"
c3d "$L" -thresh 1 inf 1 0 -o "$L"
if c3d "$L" "$O" "$B" -scale -1 -add "$T" -scale -1 -add "$B" "$T" -multiply -add -multiply "$B" "$O" "$T" -scale -1 -add -multiply -thresh 1 inf 2 0 -add "$T" -thresh 1 inf 3 0 -add -o "$SEGMENTATIONS_DIR/$CASE_NAME.nii.gz"; then
echo "$MEDRECNUM,$CASE_NUM" >> $CSV_FILE
mv "$OUTPUT_DIR/image.nii.gz" "$CASE_DIR/$CASE_CHANNEL.nii.gz"
mv "$OUTPUT_DIR/mask_Biliary.nii.gz" "$LABEL1_DIR/$CASE_NAME.nii.gz"
mv "$OUTPUT_DIR/mask_Liver.nii.gz" "$LABEL2_DIR/$CASE_NAME.nii.gz"
CASE_NUM=$((CASE_NUM+1))
else
echo "$MEDRECNUM error"
fi
done
echo "finished"



