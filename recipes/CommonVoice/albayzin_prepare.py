"""
Data preparation.

Download: https://voice.mozilla.org/en/datasets

Author
------
Wiliam Fernando López Gavilánez, 2022
"""

import os
import csv
import re
import logging
import torchaudio
import unicodedata

from tqdm import tqdm
from tqdm.contrib import tzip

# Fer
import pandas as pd

logger = logging.getLogger(__name__)


def prepare_albayzin(
    cv_data_folder,
    albayzin_data_folder,
    save_folder,
    cv_train_tsv_file=None,
    cv_dev_tsv_file=None,
    cv_test_tsv_file=None,
    albayzin_train_tsv_file=None,
    albayzin_dev_tsv_file=None,
    accented_letters=False,
    language="sp",
    skip_prep=False,
):
    """
    Prepares the csv files for the Mozilla Common Voice dataset.
    Download: https://voice.mozilla.org/en/datasets

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original Common Voice dataset is stored.
        This path should include the lang: /datasets/CommonVoice/en/
    save_folder : str
        The directory where to store the csv files.
    train_tsv_file : str, optional
        Path to the Train Common Voice .tsv file (cs)
    dev_tsv_file : str, optional
        Path to the Dev Common Voice .tsv file (cs)
    test_tsv_file : str, optional
        Path to the Test Common Voice .tsv file (cs)
    accented_letters : bool, optional
        Defines if accented letters will be kept as individual letters or
        transformed to the closest non-accented letters.
    language: str
        Specify the language for text normalization.
    skip_prep: bool
        If True, skip data preparation.

    Example
    -------
    >>> from recipes.CommonVoice.common_voice_prepare import prepare_common_voice
    >>> data_folder = '/datasets/CommonVoice/en'
    >>> save_folder = 'exp/CommonVoice_exp'
    >>> train_tsv_file = '/datasets/CommonVoice/en/train.tsv'
    >>> dev_tsv_file = '/datasets/CommonVoice/en/dev.tsv'
    >>> test_tsv_file = '/datasets/CommonVoice/en/test.tsv'
    >>> accented_letters = False
    >>> duration_threshold = 10
    >>> prepare_common_voice( \
                 data_folder, \
                 save_folder, \
                 train_tsv_file, \
                 dev_tsv_file, \
                 test_tsv_file, \
                 accented_letters, \
                 language="en" \
                 )
    """

    if skip_prep:
        return

    # CommonVoice
    if cv_train_tsv_file is None:
        cv_train_tsv_file = cv_data_folder + "/train.tsv"
    else:
        cv_train_tsv_file = cv_train_tsv_file

    if cv_dev_tsv_file is None:
        cv_dev_tsv_file = cv_data_folder + "/dev.tsv"
    else:
        cv_dev_tsv_file = cv_dev_tsv_file

    if cv_test_tsv_file is None:
        cv_test_tsv_file = cv_data_folder + "/test.tsv"
    else:
        cv_test_tsv_file = cv_test_tsv_file

    # Setting the save folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Setting ouput files
    save_csv_train = save_folder + "/train.csv"
    save_csv_dev = save_folder + "/dev.csv"
    save_csv_test = save_folder + "/test.csv"

    # If csv already exists, we skip the data preparation
    if skip(save_csv_train, save_csv_dev, save_csv_test):

        msg = "%s already exists, skipping data preparation!" % (save_csv_train)
        logger.info(msg)

        msg = "%s already exists, skipping data preparation!" % (save_csv_dev)
        logger.info(msg)

        msg = "%s already exists, skipping data preparation!" % (save_csv_test)
        logger.info(msg)

        return
    
    # Additional checks to make sure the data folder contains Common Voice
    check_commonvoice_folders(cv_data_folder)

    # Creating csv file for training data
    msg = "CommonVoice train split"
    logger.info(msg)
    cv_train_list = create_commonvoice_list(
        cv_train_tsv_file,
        cv_data_folder,
        accented_letters,
        language,
    )

    msg = "Albayzin train split"
    logger.info(msg)
    albayzin_train_list = create_albayzin_list(
        albayzin_train_tsv_file,
        albayzin_data_folder,
        accented_letters
    )
    train_list = ["ID", "duration", "wav", "spk_id", "wrd"] + cv_train_list + albayzin_train_list
    save_csv(save_csv_train, train_list)

    # Creating csv file for dev data
    msg = "CommonVoice dev split"
    logger.info(msg)
    cv_dev_list = create_commonvoice_list(
        cv_dev_tsv_file,
        cv_data_folder,
        accented_letters,
        language
    )

    msg = "Albayzin dev split"
    logger.info(msg)
    albayzin_dev_list = create_albayzin_list(
        albayzin_dev_tsv_file,
        albayzin_data_folder,
        accented_letters
    )
    dev_list = ["ID", "duration", "wav", "spk_id", "wrd"] + cv_dev_list + albayzin_dev_list
    save_csv(save_csv_dev, dev_list)

    # Creating csv file for test data
    msg = "CommonVoice test split"
    logger.info(msg)
    cv_test_list = create_commonvoice_list(
        cv_test_tsv_file,
        cv_data_folder,
        accented_letters,
        language,
    )
    test_list = ["ID", "duration", "wav", "spk_id", "wrd"] + cv_test_list
    save_csv(save_csv_test, test_list)


def save_csv(csv_file, csv_lines):
    msg = "Creating csv lists in %s ..." % (csv_file)
    logger.info(msg)
    with open(csv_file, mode="w", encoding="utf-8") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        for line in csv_lines:
            csv_writer.writerow(line)

    # Final prints
    msg = "%s successfully created!" % (csv_file)
    logger.info(msg)


def skip(save_csv_train, save_csv_dev, save_csv_test):
    """
    Detects if the Common Voice data preparation has been already done.

    If the preparation has been done, we can skip it.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """

    # Checking folders and save options
    skip = False

    if (
        os.path.isfile(save_csv_train)
        and os.path.isfile(save_csv_dev)
        and os.path.isfile(save_csv_test)
    ):
        skip = True

    return skip


def create_albayzin_list(
    orig_tsv_file, data_folder, accented_letters=False
):
    # Check if the given files exists
    if not os.path.isfile(orig_tsv_file):
        msg = "\t%s doesn't exist, verify your dataset!" % (orig_tsv_file)
        logger.info(msg)
        raise FileNotFoundError(msg)

    # We load and skip the header
    loaded_tsv = pd.read_csv(orig_tsv_file, header=0, sep='\t')

    # Drop duplicates
    loaded_tsv = loaded_tsv.drop_duplicates(subset=['Sample_ID']).reset_index(drop=True)

    nb_samples = str(len(loaded_tsv.index))

    msg = "Preparing CSV files for %s samples ..." % (str(nb_samples))
    logger.info(msg)

    csv_lines = []

    # Get clips path
    partition_name = orig_tsv_file.split('/')[-1].split('_')[0].replace('.tsv', '')
    clips_path = os.path.join(data_folder, 'segmented_clips', partition_name)

    # Start processing lines
    progress_bar = tqdm(total=len(loaded_tsv.index), desc='Albayzin')
    total_duration = 0.0
    for idx, row in loaded_tsv.iterrows():

        # Path is at indice 1 in Common Voice tsv files. And .mp3 files
        # are located in datasets/lang/clips/
        wav_path = os.path.join(clips_path, row['Sample_ID'] + '.wav')
        file_name = row['Sample_ID']
        spk_id = row['Speaker_ID']
        snt_id = file_name
        words = row['Transcription'].lower()

        # Setting torchaudio backend to sox-io (needed to read mp3 files)
        if torchaudio.get_audio_backend() != "sox_io":
            logger.warning("This recipe needs the sox-io backend of torchaudio")
            logger.warning("The torchaudio backend is changed to sox_io")
            torchaudio.set_audio_backend("sox_io")

        # Reading the signal (to retrieve duration in seconds)
        if os.path.isfile(wav_path):
            info = torchaudio.info(wav_path)
        else:
            msg = "\tError loading: %s" % (str(len(file_name)))
            logger.info(msg)
            continue

        duration = info.num_frames / info.sample_rate
        
        # Check empty samples due bad alignment
        if duration == 0.0:
            continue

        total_duration += duration

        # Unicode Normalization
        words = unicode_normalisation(words)

        # !! Language specific cleaning !!
        # Important: feel free to specify the text normalization
        # corresponding to your alphabet.

        words = re.sub(
            "[^’'A-Za-z0-9À-ÖØ-öø-ÿЀ-ӿéæœâçèàûî]+", " ", words
        ).upper()
        
        # Spanish specific cleaning
        words = words.replace("'", " ")
        words = words.replace("’", " ")

        # catalan = "ÏÀÒ"
        words = words.replace("Ï", "I")
        words = words.replace("À", "A")
        words = words.replace("Ò", "O")
        
        # Remove russian speech: Cyrillic chars
        cyrillic_chars = "\u0400-\u04FF"
        # Remove proper nouns from other languages
        proper_nouns = "ŒÛÙÅÌÞÎÝÕÆÐÖÃÄËØÊÔÂ"
    
        if(re.search("[" + cyrillic_chars  + proper_nouns + "]", words)):
            continue
        
        # Remove accents if specified
        if not accented_letters:
            words = strip_accents(words)
            words = words.replace("'", " ")
            words = words.replace("’", " ")

        # Remove multiple spaces
        words = re.sub(" +", " ", words)

        # Remove spaces at the beginning and the end of the sentence
        words = words.lstrip().rstrip()

        # Getting chars
        chars = words.replace(" ", "_")
        chars = " ".join([char for char in chars][:])

        # Remove too short sentences (or empty):
        if len(words.split(" ")) < 3:
            continue

        # Composition of the csv_line
        csv_line = [snt_id, str(duration), wav_path, spk_id, str(words)]

        # Adding this line to the csv_lines list
        csv_lines.append(csv_line)

        progress_bar.update(1)

    # Final prints
    msg = "Number of samples: %s " % (str(len(nb_samples)))
    logger.info(msg)
    msg = "Total duration: %s Hours" % (str(round(total_duration / 3600, 2)))
    logger.info(msg)

    return csv_lines


def create_commonvoice_list(
    orig_tsv_file, data_folder, accented_letters=False, language="sp"
):
    """
    Creates the csv file given a list of wav files.

    Arguments
    ---------
    orig_tsv_file : str
        Path to the Common Voice tsv file (standard file).
    data_folder : str
        Path of the CommonVoice dataset.
    accented_letters : bool, optional
        Defines if accented letters will be kept as individual letters or
        transformed to the closest non-accented letters.

    Returns
    -------
    None
    """

    # Check if the given files exists
    if not os.path.isfile(orig_tsv_file):
        msg = "\t%s doesn't exist, verify your dataset!" % (orig_tsv_file)
        logger.info(msg)
        raise FileNotFoundError(msg)

    # We load and skip the header
    loaded_csv = open(orig_tsv_file, "r").readlines()[1:]
    nb_samples = str(len(loaded_csv))

    msg = "Preparing CSV files for %s samples ..." % (str(nb_samples))
    logger.info(msg)

    csv_lines = []

    # Start processing lines
    total_duration = 0.0
    for line in tzip(loaded_csv):

        line = line[0]

        # Path is at indice 1 in Common Voice tsv files. And .mp3 files
        # are located in datasets/lang/clips/
        mp3_path = data_folder + "/clips/" + line.split("\t")[1]
        file_name = mp3_path.split(".")[-2].split("/")[-1]
        spk_id = line.split("\t")[0]
        snt_id = file_name

        # Setting torchaudio backend to sox-io (needed to read mp3 files)
        if torchaudio.get_audio_backend() != "sox_io":
            logger.warning("This recipe needs the sox-io backend of torchaudio")
            logger.warning("The torchaudio backend is changed to sox_io")
            torchaudio.set_audio_backend("sox_io")

        # Reading the signal (to retrieve duration in seconds)
        if os.path.isfile(mp3_path):
            info = torchaudio.info(mp3_path)
        else:
            msg = "\tError loading: %s" % (str(len(file_name)))
            logger.info(msg)
            continue

        duration = info.num_frames / info.sample_rate
        total_duration += duration

        # Getting transcript
        words = line.split("\t")[2]

        # Unicode Normalization
        words = unicode_normalisation(words)

        # !! Language specific cleaning !!
        # Important: feel free to specify the text normalization
        # corresponding to your alphabet.

        if language in ["en", "fr", "it", "rw", "sp"]:
            words = re.sub(
                "[^’'A-Za-z0-9À-ÖØ-öø-ÿЀ-ӿéæœâçèàûî]+", " ", words
            ).upper()

        if language == "fr":
            # Replace J'y D'hui etc by J_ D_hui
            words = words.replace("'", " ")
            words = words.replace("’", " ")

        elif language == "ar":
            HAMZA = "\u0621"
            ALEF_MADDA = "\u0622"
            ALEF_HAMZA_ABOVE = "\u0623"
            letters = (
                "ابتةثجحخدذرزسشصضطظعغفقكلمنهويىءآأؤإئ"
                + HAMZA
                + ALEF_MADDA
                + ALEF_HAMZA_ABOVE
            )
            words = re.sub("[^" + letters + " ]+", "", words).upper()
        
        elif language == "ga-IE":
            # Irish lower() is complicated, but upper() is nondeterministic, so use lowercase
            def pfxuc(a):
                return len(a) >= 2 and a[0] in "tn" and a[1] in "AEIOUÁÉÍÓÚ"

            def galc(w):
                return w.lower() if not pfxuc(w) else w[0] + "-" + w[1:].lower()

            words = re.sub("[^-A-Za-z'ÁÉÍÓÚáéíóú]+", " ", words)
            words = " ".join(map(galc, words.split(" ")))
        
        elif language == "sp":
            words = words.replace("'", " ")
            words = words.replace("’", " ")

            # catalan = "ÏÀÒ"
            words = words.replace("Ï", "I")
            words = words.replace("À", "A")
            words = words.replace("Ò", "O")
            
            # Remove russian speech: Cyrillic chars
            cyrillic_chars = "\u0400-\u04FF"
            # Remove proper nouns from other languages
            proper_nouns = "ŒÛÙÅÌÞÎÝÕÆÐÖÃÄËØÊÔÂ"
       
            if(re.search("[" + cyrillic_chars  + proper_nouns + "]", words)):
                continue
        
        # Remove accents if specified
        if not accented_letters:
            words = strip_accents(words)
            words = words.replace("'", " ")
            words = words.replace("’", " ")

        # Remove multiple spaces
        words = re.sub(" +", " ", words)

        # Remove spaces at the beginning and the end of the sentence
        words = words.lstrip().rstrip()

        # Getting chars
        chars = words.replace(" ", "_")
        chars = " ".join([char for char in chars][:])

        # Remove too short sentences (or empty):
        if len(words.split(" ")) < 3:
            continue

        # Composition of the csv_line
        csv_line = [snt_id, str(duration), mp3_path, spk_id, str(words)]

        # Adding this line to the csv_lines list
        csv_lines.append(csv_line)

    # Prints
    msg = "Number of samples: %s " % (str(len(loaded_csv)))
    logger.info(msg)
    msg = "Total duration: %s Hours" % (str(round(total_duration / 3600, 2)))
    logger.info(msg)

    return csv_lines


def check_commonvoice_folders(data_folder):
    """
    Check if the data folder actually contains the Common Voice dataset.

    If not, raises an error.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If data folder doesn't contain Common Voice dataset.
    """

    files_str = "/clips"

    # Checking clips
    if not os.path.exists(data_folder + files_str):

        err_msg = (
            "the folder %s does not exist (it is expected in "
            "the Common Voice dataset)" % (data_folder + files_str)
        )
        raise FileNotFoundError(err_msg)


def unicode_normalisation(text):

    try:
        text = unicode(text, "utf-8")
    except NameError:  # unicode is a default on python 3
        pass
    return str(text)


def strip_accents(text):

    text = (
        unicodedata.normalize("NFD", text)
        .encode("ascii", "ignore")
        .decode("utf-8")
    )

    return str(text)
