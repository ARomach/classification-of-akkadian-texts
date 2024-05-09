# Genre, Period, and Provenience Classification in Akkadian Cuneiform Documents

This repository includes data and code for the first attempt to perform genre, period, and provenience classification in Akkadian cuneiform documents. I use two baseline models, Naive Bayes and Logistic Regression, and three BERT models fine-tuned to this task. Each model and classification task was trained and tested on four versions of the same Akkadian texts: lemmatized, normalized (phonetically reconstructed), segmented Unicode cuneiform, and unsegmented Unicode cuneiform. The best performing models for each classification task are multilingual BERT with normalization for genre (96% weighted F1), Arabic BERT with segmented Unicode cuneiform for period (97%), and multilingual BERT with normalization for provenience (93%).

## Environment setup

The `requirements.txt` includes a list of all dependencies and libraries used with specific versions.

All the code is in Jupyter notebooks. After environment is set up, the code can be run as is, see more instructions in the `Main` sections for possible adaptations. The BERT-transformer notebook was only run in colab, and has optional sections for colab environment.

## Dataset description

The dataset used in this study was taken from the Open Richly Annotated Cuneiform Corpus ([ORACC](https://oracc.museum.upenn.edu/projectlist.html)). ORACC is subdivided into projects that can be downloaded individually. Instruction and code on how to access and parse ORACC projects is available on the [Computational Assyriology project](https://github.com/niekveldhuis/compass/tree/master/2_1_Data_Acquisition_ORACC). Most of the corpus is in the Akkadian or Sumerian languages. While the entire corpus as currently downloaded holds 114,967 texts, after removing non-Akkadian texts (97,714 texts) and removing texts whose length is below 20 words (7,969 texts) or over 2000 words (15 texts), and removing duplicate entries, the size of the remaining corpus is 9,269 texts and 1,034,737 words.

Metadata on each project in ORACC--which include genre, period, and provenience--was inconsistent. Using [OpenRefine](https://openrefine.org/), classes in each of the three categories were converged based on the author's familiarity with the corpus. The final dataset is organized in a CSV file with the following columns:

- `index`: the unique identifier of each text.
- `project`: the name of the project from which the text was taken. Can be compared to the full list of projects found at the [ORACC project list](https://oracc.museum.upenn.edu/projectlist.html).
- `genre`: the original genre classification from the project's metadata, that was used to create the broad genre categories for this study.
- `supergenre_160424`: the broad genre categories used in this study.
- `period`: the original period classification from the project's metadata, that was used to create the broad period categories for this study.
- `superperiod_160424`: the broad period categories used in this study.
- `provenience`: the original provenience classification from the project's metadata, that was used to create the broad provenience categories for this study.
- `superprovenience_160424`: the broad provenience categories used in this study.
- `language`: a column from the original metadata that did not exist in all ORACC projects. Inconsistencies were found between this and the `dialect` column and it is unused for now.
- `Dialect`: the ORACC code points for dialect, as found in the original metadata. A closer examination found inconsistencies between this and the `language` column and between the dialects known to be in some of the projects' texts. This variable is unused for now.
- `lemm`: the lemmatized version of the text.
- `lemm_length_full`: the full length of the lemmatized text, including UNK and X tokens.
- `lemm_length_partial`: the partial length of the lemmatized text, excluding UNK and X tokens.
- `norm`: the normalized version of the text.
- `norm_length_full`: the full length of the normalized text, including UNK and X tokens.
- `norm_length_partial`: the partial length of the normalized text, excluding UNK and X tokens.
- `seg_uni`: the segmented Unicode cuneiform version of the text.
- `seg_uni_length_full`: the full length of the segmented Unicode cuneiform text, including X tokens.
- `seg_uni_length_partial`: the partial length of the segmented Unicode cuneiform text, excluding X tokens.
- `unseg_uni`: the unsegmented Unicode cuneiform version of the text.
- `unseg_uni_length_full`: the full length of the unsegmented Unicode cuneiform text, including X tokens. The same as `seg_uni_length_full`.
- `unseg_uni_length_partial`: the partial length of the unsegmented Unicode cuneiform text, excluding X tokens. The same as `seg_uni_length_partial`
- `lemm_ratio`: the ratio between `lemm_length_partial` and `lemm_length_full`
- `norm_ratio`: the ratio between `norm_length_partial` and `norm_length_full`
- `uni_ratio`: the ratio between `seg_uni_length_partial` and `seg_uni_length_full`

As the dataset was merged from separate projects, there were some textual duplicates, identified through their unique `index`. Duplicates were removed automatically by keeping the version whose `lemm_length_partial`, `norm_length_partial`, and `seg_uni_length_partial` were largest.

The classes in each category are defined as follows:

1. Genre
   - `Archival`: ephemeral documents that are primarily judicial or administrative in nature. 
   - `Letter`: all types of letters sent from one person to another, be they private individuals or state and royal correspondences.
   - `Scientific`: all documents related to Mesopotamian scholarly production. As much as possible, this follows ancient definition of what the Mesopotamian themselves saw as scholarship, which includes religious texts. As such, this class encapsulates texts that are usually viewed as distinct in cuneiform studies, such as lexical lists, omens, astronomical observations, rituals, hymns, and incantations.
   - `Royal/Monumental`: royal or monumental inscriptions. These are usually either long narrative accounts describing the great deeds of kings, or short dedicatory inscriptions on important or valuable objects. 
   - `Literary`: literary texts, narrowly defined as texts that have a purpose only as literature. Stylistically, there is overlap with certain rituals and hymns that sometimes use literary language.
2. Period
   - `2ndMill-1stHalf`: the first half of the second millennium BCE, approximately 2000-1600 BCE. This period is primarily politically defined by Amorite kingdoms ruling Babylonian city-states, sometimes expanding to Assyria and further to the west.
   - `2ndMill-2ndHalf` the second half of the second millennium BCE, approximately 1600-1100 BCE. This period is primarily politically defined by the rise of competing kingdoms throughout the ancient Near East from Babylonia to Egypt, who kept diplomatic relations alongside warfare, imperial conquests, and puppet and vassal rulers.
   - `Neo-Assyrian; Neo-Babylonian; Achaemenid`: also known as the age of empires, this period is from approximately 1100 BCE until the conquest of Alexander the Great in 331 BCE. It is defined by the direct continuation from one empire to another (Neo-Assyrian to Neo-Babylonian and Neo-Babylonian to Achaemenid/Persian) ruling over the majority of the ancient Near East.
   - `Hellenistic onwards`: the period from Alexander's conquest in 331 BCE until the end of the use of the cuneiform script in the first centuries CE. At this time, Akkadian was no longer a spoken language, but it was still used in temples which preserved their age-long traditions and by arcane administrative and legal institutions. The Hellenistic conquest marks a common break off point compared to earlier periods, despite cultural continuities within groups who still used the cuneiform script.
   - `First Millennium`: texts that cannot be assigned to either `Neo-Assyrian; Neo-Babylonian; Achaemenid` or `Hellenistic onwards` based on the available metadata or current knowledge. These are often texts that are considered part of the stream of tradition, scientific or literary texts which were copied continueously, therefore the exact date of writing for a specific document is hard to detect from a linguistic and paleographic perspective. Texts of this class were removed before period classification.
   - `Unknown`: texts whose date of composition is entirely unknown. These were removed before period classification.
3. Provenience
   - `Assyria`: texts from the Assyrian heartland, defined geographically by the three cities that mark its triangular boundaries in modern day northern Iraq on the Tigris river and its tributaries: Assur (mod. Qal'at Sherqat), Arbela (mod. Erbil), and Nineveh (mod. Mosul). 
   - `Babylonia`: texts from Babylonia proper, i.e. the territory primarily between the Tigris and Euphrates rivers until the area of modern day Baghdad.
   - `West`: texts from the west of Mesopotamia, i.e. west of Assyria and Babylonia. This includes texts found at sites in proximity to the Euphrates river and its tributaries north of Babylonia, as well as Anatolia, the Levant, and Egypt.
   - `East`: texts found to the east of Mesopotamia, i.e. east of Assyria and Babylonia. This is primarily texts from modern day Iran, ancient Elam and Persia. As only a limited number of these texts are in the dataset, this class was removed before provenience classification.
   - `Unknown`: texts whose place of composition is entirely unknown. These were removed before provenience classification.
