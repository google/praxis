#!/bin/bash

# This script prepare a new release by:
# 1) update version number in setup.py and cloudbuild-release.yaml
# 2) add a new section in RELEASE.md with version and corresponding commit

set -e -x

function print_help_and_exit {
 echo "Usage: prepare_release.sh -x <praxis_version> -d <build_date:YYYYMMDD> "
 echo "exp: bash prepare_release.sh -x 0.2.0 -d 20221114"
 exit 0
}

while getopts "hd:x:" opt; do
  case $opt in
    x)
      PRAXIS_VERSION=${OPTARG}
      ;;
    d)
      BUILD_DATE=${OPTARG}
      ;;
    *)
      print_help_and_exit
      ;;
  esac
done

RELEASE_NOTE="../RELEASE.md"
RELEASE_NOTE_NEW="release_new.md"

if [[ -z "$BUILD_DATE" ]]; then
  echo "Build date is required!"
  exit 1
fi

if [[ -z "$PRAXIS_VERSION" ]]; then
  echo "praxis version is required!"
  exit 1
fi

echo "Build date: "$BUILD_DATE
echo "PRAXIS version: "$PRAXIS_VERSION

sed -i "s/version='[0-9.]*'/version='$PRAXIS_VERSION'/" setup.py
sed -i "s/_RELEASE_VERSION: '[0-9.]*'/_RELEASE_VERSION: '$PRAXIS_VERSION'/" cloudbuild-release.yaml
gsutil cp gs://pax-on-cloud-tpu-project/wheels/"$BUILD_DATE"/praxis_commit.txt ./
PRAXIS_COMMIT=$(<praxis_commit.txt)
rm praxis_commit.txt
echo "PRAXIS_COMMIT: " $PRAXIS_COMMIT
[ -e $RELEASE_NOTE_NEW ] && rm $RELEASE_NOTE_NEW
echo "# Version: $PRAXIS_VERSION" >> $RELEASE_NOTE_NEW
echo "## Major Features and Improvements" >> $RELEASE_NOTE_NEW
echo "## Breaking changes" >> $RELEASE_NOTE_NEW
echo "## Deprecations" >> $RELEASE_NOTE_NEW
echo "## Note" >> $RELEASE_NOTE_NEW
echo "*   Version: $PRAXIS_VERSION" >> $RELEASE_NOTE_NEW
echo "*   Build Date: $BUILD_DATE" >> $RELEASE_NOTE_NEW
echo "*   Praxis commit: $PRAXIS_COMMIT" >> $RELEASE_NOTE_NEW
RELEASE_NOTE_TMP="RELEASE.tmp.md"
cat $RELEASE_NOTE_NEW $RELEASE_NOTE >> $RELEASE_NOTE_TMP
rm $RELEASE_NOTE_NEW
rm $RELEASE_NOTE
mv $RELEASE_NOTE_TMP $RELEASE_NOTE
