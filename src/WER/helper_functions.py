# insert helper functions from aphasia_transcription_cleaning_analysis here
from whisper_normalizer.english import EnglishTextNormalizer
from cmath import isnan
import pandas as pd
import numpy as np
import os
import re
from nltk.tokenize import word_tokenize
from jiwer import wer
from num2words import num2words
pd.options.display.max_rows = 100000000
pd.options.display.max_colwidth = None
english_normalizer = EnglishTextNormalizer()


def remove_consecutive_words(input_string):
    if not input_string:  # Check if the input string is empty
        return ""
    words = input_string.split()

    # fix the index out of range error
    if len(words) == 1:
        return input_string

    # check the index out of range error
    if len(words) == 2:
        if words[0] == words[1]:
            return words[0]
        else:
            return input_string

    if len(words) == 0:
        return ""
    # check if words[0] actually exists
    result = [words[0]]

    for word in words[1:]:
        if word != result[-1]:
            result.append(word)

    return ' '.join(result)


def modify_edgecase_value(df, segment_name, column_name, new_value):
    # if the column segment_name exists in the dataframe
    df.loc[df['segment_name'] == segment_name, column_name] = new_value


    return None


filler_words = ['um', 'umm', 'uh', 'mhm', 'mm', 'ugh', 'uhhuh', 'mm-hmm',
                "uhh", "mmhmm", "uh-huh", "uh-hmm", "uh-hm", "uh-hm", "hm", "hmm", "emmm"]

states = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY',
}



firstname_dict = {
    'Ashley': 'FirstnameA', 'Annette': 'FirstnameA', 'Alice': 'FirstnameA', 'Andrew': 'FirstnameA',
    'Anthony': 'FirstnameA', 'Arthur': 'FirstnameA', 'Atta': 'FirstnameA', 'Alex': 'FirstnameA',
    'Art': 'FirstnameA', 'Alicia': 'FirstnameA', 'Ali': 'FirstnameA', 'Alisha': 'FirstnameA',
    'Alijah': 'FirstnameA', 'Adolph': 'FirstnameA', 'Ashton': 'FirstnameA',
    'Beverly': 'FirstnameB', 'Barbara': 'FirstnameB', 'Brie': 'FirstnameB', 'Bree': 'FirstnameB',
    'Billy': 'FirstnameB', 'Bonnie': 'FirstnameB', 'Bonny': 'FirstnameB', 'Bo': 'FirstnameB',
    'Ben': 'FirstnameB',
    "Catherine":"FirstnameC", "Katherine":"FirstnameC",
    'Chris': 'FirstnameC', 'Connie': 'FirstnameC', 'Cony': 'FirstnameC', 'Cameron': 'FirstnameC',
    'Corey': 'FirstnameC', 'Chloe': 'FirstnameC', 'Christine': 'FirstnameC', 'Charlie': 'FirstnameC',
    'Christian': 'FirstnameC', 'Chrissy': 'FirstnameC', 'Cynthia': 'FirstnameC', 'Corbin': 'FirstnameC',
    'Corbyn': 'FirstnameC',
    'Dean': 'FirstnameD', 'Dan': 'FirstnameD', 'Debbie': 'FirstnameD', 'Donna': 'FirstnameD',
    'Dave': 'FirstnameD', 'David': 'FirstnameD', 'Doug': 'FirstnameD', 'Don': 'FirstnameD',
    'Dick': 'FirstnameD', 'Debby': 'FirstnameD', 'Diane': 'FirstnameD', 'Denise': 'FirstnameD',
    'Dana': 'FirstnameD', 'Deny': 'FirstnameD', 'Danny': 'FirstnameD', 'Derrickman': 'FirstnameD',
    'Derek': 'FirstnameD', 'Derriken': 'FirstnameD', 'Derrickin': 'FirstnameD', 'Danielle': 'FirstnameD',
    'Dennis': 'FirstnameD', 'Denasia': 'FirstnameD', 'Danasia': 'FirstnameD', 'Dale': 'FirstnameD',
    'Dawn': 'FirstnameD', 'Devon': 'FirstnameD', 'Devin': 'FirstnameD',
    'Ed': 'FirstnameE', 'Earl': 'FirstnameE', 'Elaine': 'FirstnameE', 'Elizabeth': 'FirstnameE',
    'Emma': 'FirstnameE',
    'Ginger': 'FirstnameG', 'Grace': 'FirstnameG', 'Gerald': 'FirstnameG', 'George': 'FirstnameG',
    'Gay': 'FirstnameG', 'Greg': 'FirstnameG',
    'Harris': 'FirstnameH', 'Helen': 'FirstnameH', 'Harry': 'FirstnameH',
    'Jackie': 'FirstnameJ', 'Jack': 'FirstnameJ', 'Joey': 'FirstnameJ', 'Jeff': 'FirstnameJ',
    'June': 'FirstnameJ', 'John': 'FirstnameJ', 'Jerry': 'FirstnameJ', 'Jimmy': 'FirstnameJ',
    'Jamie': 'FirstnameJ', 'Jim': 'FirstnameJ', 'Joy': 'FirstnameJ', 'Jason': 'FirstnameJ',
    'Joe': 'FirstnameJ', 'Jennifer': 'FirstnameJ', 'Janice': 'FirstnameJ', 'Janis': 'FirstnameJ',
    'Jean': 'FirstnameJ', 'Joan': 'FirstnameJ',
    'Karen': 'FirstnameK', 'Karina': 'FirstnameK', 'Kayla': 'FirstnameK', 'Katie': 'FirstnameK',
    'Kessler': 'FirstnameK', 'Kestler': 'FirstnameK', 'Kenny': 'FirstnameK', 'Carren': 'FirstnameK',
    'Lorraine': 'FirstnameL', 'Lauraine': 'FirstnameL', 'Linda': 'FirstnameL', 'Lory': 'FirstnameL',
    'Laurie': 'FirstnameL', 'Lori': 'FirstnameL', 'Lee': 'FirstnameL', 'Lisa': 'FirstnameL',
    'Laura': 'FirstnameL', 'Lara': 'FirstnameL',
    'Mary': 'FirstnameM', 'Matt': 'FirstnameM', 'Maria': 'FirstnameM', 'Melissa': 'FirstnameM',
    'Margaret': 'FirstnameM', 'Marilyn': 'FirstnameM', 'Margo': 'FirstnameM', 'Margot': 'FirstnameM',
    'Max': 'FirstnameM', 'Madison': 'FirstnameM', 'Marie': 'FirstnameM', 'Murray': 'FirstnameM',
    'Michael': 'FirstnameM', 'Mike': 'FirstnameM', 'Maureen': 'FirstnameM', 'Mourin': 'FirstnameM',
    'Mourine': 'FirstnameM', 'Mikoto': 'FirstnameM', 'Makoto': 'FirstnameM', 'Maco': 'FirstnameM',
    'Maddie': 'FirstnameM', 'Mattie': 'FirstnameM', 'Madeline': 'FirstnameM', 'Madeleine': 'FirstnameM',
    'Mel': 'FirstnameM',
    'Nancy': 'FirstnameN', 'Nadia': 'FirstnameN',
    'Polly': 'FirstnameP', 'Phil': 'FirstnameP', 'Pam': 'FirstnameP', 'Patty': 'FirstnameP',
    'Piper': 'FirstnameP', 'Pia': 'FirstnameP', 'Pat': 'FirstnameP',
    'Ryan': 'FirstnameR', 'Ron': 'Firstname', 'Ro':"Firstname","Rowe":"Firstname",
    'STANDARD_firstname':"firstname",
}
lastname_dict = {
    'Adams': 'LastnameA',
    'Bridewell': 'LastnameB', 'Bridwell': 'LastnameB', 'Bernstein': 'LastnameB', 'Birnbaum': 'LastnameB', 
    'Barell': 'LastnameB', 'Burrell': 'LastnameB', 'Barr': 'LastnameB', 'Bar': 'LastnameB', 'Burr': 'LastnameB', 
    'Bauer': 'LastnameB', 'Bower': 'LastnameB', 'Bernie': 'LastnameB', 'Burney': 'LastnameB', 'Verney': 'LastnameB', 
    'Bernay': 'LastnameB', 'Barter': 'LastnameB', 'Banter': 'LastnameB', 'Butter': 'LastnameB', 'Brock': 'LastnameB', 
    'Bowman': 'LastnameB', 'Bradel': 'LastnameB', 'Brittle': 'LastnameB', 'Bridle': 'LastnameB', 
    'Cater': 'LastnameC', 'Kater': 'LastnameC',
    'Dulski': 'LastnameD', 'Dolowski': 'LastnameD', 'Doloski': 'LastnameD', 'Delosky': 'LastnameD',
    'Griffith': 'LastnameG', 'Griffit': 'LastnameG', 'Green': 'LastnameG',
    'Jacobs': 'LastnameJ',
    'Lancaster': 'LastnameL', 'Lankaster': 'LastnameL', 'LaBelle': 'LastnameL', 'Label': 'LastnameL',
    'McCormick': 'LastnameM', 'Mccormick': 'LastnameM', 'McCorick': 'LastnameM',
    'O’Gara': 'LastnameO', 'Ogara': 'LastnameO',
    'Rupp': 'LastnameR',
    'Sobol': 'LastnameS', 'Sobel': 'LastnameS', 'Sobal': 'LastnameS', 'Soville': 'LastnameS',
    'Wilson': 'LastnameW',
    'Young': 'LastnameY',
    "STANDARD_lastname":"lastname"
}

def standardize_groundtruth_names(text):
    '''
    Standardize names in the input text.
    '''
    tokens = text.split()
    for i, word in enumerate(tokens):
        if re.match(r'firstname', tokens[i].lower()):
            tokens[i] = "firstname"
        if re.match(r'lastname', tokens[i].lower()):
            tokens[i] = "lastname"
    return " ".join(tokens)

def standardize_asr_names(text):
    '''
    Standardize names in the input text.
    '''
    tokens = text.split()
    for i, word in enumerate(tokens):
        if firstname_dict.get(re.sub(r'[^\w\s]', '', tokens[i].capitalize())):
            tokens[i] = firstname_dict['STANDARD_firstname']
        if lastname_dict.get(re.sub(r'[^\w\s]', '', tokens[i].capitalize())):
            tokens[i] = lastname_dict['STANDARD_lastname']
    return " ".join(tokens)

def fix_state_abbrevs(text):
    # Standardize state abbreviations
    ix = 0
    state_result = []
    if text is None:
        return text
    wordlist = text.split()
    while ix < len(wordlist):
        word = wordlist[ix].lower().capitalize()
        if word in states.keys():  # is this correct check?
            new_word = states[word]
        elif (ix < len(wordlist)-1) and ((word + ' ' + wordlist[ix+1].lower().capitalize()) in states.keys()):
            new_word = states[(
                word + ' ' + wordlist[ix+1].lower().capitalize())]
            ix += 1
        else:
            new_word = word
        state_result.append(new_word)
        ix += 1
    text = ' '.join(state_result)
    return text


def remove_trailing_punctuations(text):
    # Define a regular expression pattern to match trailing punctuations
    trailing_punctuation_pattern = r'(\w+)([.,!?;:"]+)(\s|$)'

    # Use re.sub() to replace trailing punctuations with just the word part
    cleaned_text = re.sub(trailing_punctuation_pattern, r'\1\3', text)

    return cleaned_text


def spelling_rematch(text):
    """
    rematch spelling for certain words/phrases
    """
    pre_post_dict = {"AM": "a.m.",
                     # colloquial words
                     "buncha": "bunch of",
                     "sorta": "sort of",
                     "kinda": "kind of",
                     "hafta": "have to",
                     "hadta": "had to",
                     "hasta": "has to",
                     "useta": "used to",
                     "outta": "out of",
                     "needta": "need to",
                     "hurtcha": "hurt you",
                     "gotcha": "got you",
                     "donno": "don't know",
                     "dunno": "don't know",
                     "whaddaya": "what do you",
                     "whaddya": "what do you",
                     "whadya": "what do you",

                     "outa": "out of",
                     "growin": "growing",
                     "cuz": "cause",
                     "ok": "okay",
                     "bout": "about",
                     "coulda": "could have",
                     "shoulda": "should have",
                     "woulda": "would have",
                     "mighta": "might have",
                     "musta": "must have",
                     "gotta": "got to",
                     "needa": "need to",
                     "lotta": "lot of",
                     "oughta": "ought to",
                     "gimme": "give me",
                     "lemme": "let me",
                     "wassup": "what's up",
                     "sposta": "supposed to",
                     "supposta": "supposed to",
                     "whyntcha": "why didn't you",
                     "whatchacallit": "what do you call it",
                     "whattya": "what do you",
                     "whatyacallit": "what do you call it",
                     "ahhah": "ah ha",
                     "aha": "ah ha",
                     "alrightie": "all righty",
                     "doke": "dokie",
                     "dokey": "dokie",
                     "fella":"fellow",
                     "yup":"yeah",
                     "yep":"yeah",
                     "yack":"yak",
                     "ya": "you",

                     # unique words in this dataset
                     "12345th": "1 2 3 4 5th",
                     "abc":"a b c",
                     "abcd": "a b c d",
                     "abcde": "a b c d e",
                     "abcdefg": "a b c d e f g",
                     "abcdefghi": "a b c d e f g h i",
                     "abcdesghij":"a b c d e s g h i j",
                     "asu": "a s u",
                     "md": "m d",
                     "ip": "i p",
                     "ipa": "i p a",
                     "mdipa": "m d i p a",
                     "otpt": "o t p t",
                     "ot": "o t",
                     "pt": "p t",
                     "er": "e r",
                     "ft": "feet",
                     "dc": "d c",
                     "unc": "u n c",
                     "kilometers": "km",
                     "ohh": "oh",
                     "ooh":"oh",
                     "ohhh": "oh",
                     "ohhhh": "oh",
                     "etcetera": "et cetera",
                     "etc": "et cetera",

                     # expand abbreviations
                     "alright":"all right",
                     "aftereffect": "after effect",
                     "aircast": "air cast",
                     "angioplastic": "angio plastic",
                     "antinausea": "anti nausea",
                     "beercade": "beer cade",
                     "breadbox": "bread box",
                     "byebye": "bye bye",
                     "bookbag": "book bag",
                     "brushteeth": "brush teeth",
                     "busstop": "bus stop",
                     "birdbath": "bird bath",
                     "collarbone": "collar bone",
                     "cartwheels": "cart wheels",
                     "dreamlike": "dream like",
                     "disneylike": "disney like",
                     "exboyfriend": "ex boyfriend",
                     "exwife": "ex wife",
                     "firetrucks": "fire trucks",
                     "glowsticks": "glow sticks",
                     "homeschooled": "home schooled",
                     "icebox": "ice box",
                     "lifesaving": "life saving",
                     "lifevest": "life vest",
                     "meadowbrook": "meadow brook",
                     "ministrokes": "mini strokes",
                     "neuroconference": "neuro conference",
                     "neverland": "never land",
                     "nittypicky": "nitty picky",
                     "northbrook": "north brook",
                     "overfloating": "over floating",
                     "penpal": "pen pal",
                     "phonebook": "phone book",
                     "treehouse": "tree house",
                     "tunafish": "tuna fish",
                     "stepsisters": "step sisters",
                     "stepmother": "step mother",
                     "stepmom": "step mom",
                     "stepdad": "step dad",
                     "stepdaughters": "step daughters",
                     "stepparents": "step parents",
                     "cleanup": "clean up",
                     "premade": "pre made",
                     "streetname": "street name",
                     "lifejacket": "life jacket",
                     "lifejackets": "life jackets",
                     "lifepreserver": "life preserver",
                     "getup": "get up",
                     "fishmarket": "fish market",
                     "reexplain": "re explain",
                     "rollerskating": "roller skating",
                     "cindarella": "cinderella",
                     "warmups": "warm ups",
                     "walgreen": "walgreens",
                     "followup": "follow up",
                     "backseat": "back seat",
                     "cleanup": "clean up",
                     "buildup": "build up",
                     "postdated": "post dated",
                     "birthdate": "birth date",
                     "windowsill": "window sill",
                     "powerpuff": "power puff",
                     "glowstick": "glow stick",
                     "bedbugs": "bed bugs",
                     "exhusband": "ex husband",
                     "tollbooth": "toll booth",
                     "aftereffects": "after effects",
                     "coownership": "co ownership",
                     "stepgirls": "step girls",
                     "whitewater": "white water",
                     "motorhome": "motor home",
                     "semibeautiful": "semi beautiful",
                     "homeschooling": "home schooling",
                     "brokenhearted": "broken hearted",
                     "icecream": "ice cream",
                     "jumprope": "jump rope",
                     "roadkill":"road kill",
                     "rowboat":"row boat",
                     "spiderman": "spider man",
                     "stepgrandchildren": "step grandchildren",
                     "secondhand": "second hand",
                     "stepfather": "step father",
                     "stepmama": "step mama",
                     "storyline": "story line",
                     "seacoast": "sea coast",
                     "uhoh": "uh oh",
                     "tweettweet": "tweet tweet",
                     "somethings": "some things",
                     "washwoman": "wash woman",
                     "workday": "work day",
                     "wristwatch": "wrist watch",
                     "whatchamacallit":"what you might call it",

                     # spelling correction
                    "arururound": "around",
                     "authorative": "authoritative",
                     "accidently": "accidentally",
                     "acrosst": "across",
                     "blacktopped": "black topped",
                     "cemetary": "cemetery",
                     "chrissakes": "christ sake",
                     "Cinderalla": "cinderella",
                     "cinderelli": "cinderella",
                     "cinderalla": "cinderella",
                     "crinderella": "cinderella",
                     "cinderen": "cinderella",
                     "cinderfella": "cinderella",
                     "cinnerella": "cinderella",
                     "dads": "dad's",
                     "everythings": "everything",
                     "dreamt": "dreamed",
                     "fanny": "funny",
                     "fascitis": "fasciitis",
                     "firemens": "firemen",
                     "firetrtruck": "fire truck",
                     "gospital": "hospital",
                     "girlie": "girly",
                     "jeez": "geez",
                     "horseses": "horses",
                     "iliinois": "illinois",
                     "imemediately": "immediately",
                     "kingdomdom": "kingdom",
                     "learnt": "learned",
                     "mum": "mom",
                     "nowheres": "nowhere",
                     "nosedive": "nose dive",
                     "nonscientific": "non scientific",
                     "okey": "okay",
                     "refrigerater": "refrigerator",
                     "rerelease": "re release",
                     "retirment": "retirement",
                     "sktressed": "stressed",
                     "sepmother": "stepmother",
                     "septetember": "september",
                     "stepsissers": "step sisters",
                     "steptersisters":"step sisters",
                     "sumbrella": "umbrella",
                     "somewheres": "somewhere",
                     "supposably": "supposedly",
                     "suppuposedly": "supposedly",
                     "sublevels": "sub levels",
                     "sandcastle": "sand castle",
                     "storyteller": "story teller",
                     "stoplights": "stop lights",
                     "sixtyish": "sixty ish",
                     "springtime": "spring time",
                     "schoolwork": "school work",
                     "sonofabitch": "son of a bitch",
                     "tetherball": "tether ball",
                     "tennesee": "tennessee",
                     "themself": "themselves",
                     "tornados": "tornadoes",
                     "unbreller": "umbrella",
                     "umbrelly": "umbrella",
                     "umbreller": "umbrella",
                     "umbella": "umbrella",
                     "unbrella": "umbrella",
                     "okaydokie": "okay dokie",
                     "probly": "probably",
                     "weeklong": "week long",
                     "whippersnapper":"whipper snapper",
                     "wackadoodles":"wacka doodles"
                     }

    pre_post_dict2 = {
                    # correct abbreviations
                    "bitter sweet": "bittersweet",
                    "clothesline": "clothes line",
                    "camp site": "campsite",
                    "camp ground": "campground",
                    "cheer leader": "cheerleader",
                    "care free": "carefree",
                    "dead headed": "deadheaded",
                    "dead head": "deadhead",
                    "home care": "homecare",
                    "fire truck": "firetruck",
                    "home owners": "homeowners",
                    "proof readers": "proofreaders",
                    "proof reading":"proofreading",
                    "non verbal": "nonverbal",
                    "road blocks": "roadblocks",
                    "sour dough": "sourdough",
                    "cell phones": "cellphones",
                    "life belt": "lifebelt",
                    "life boat":"lifeboat",
                    "low life": "lowlife",
                    "face wise": "facewise",
                    "seat belt": "seatbelt",
                    "quint essential":"quintessential",
                    "neuro myasthenia": "neuromyasthenia",
                    "snow storm":"snowstorm",
                    # "is i": "isi",
                    "non verbally": "nonverbally",
                    "non stop":"nonstop",
                    "off hand": "offhand",
                    "sales person": "salesperson",
                    "a m":"am",
                    "p m":"pm",
                    "ti a": "tia",
                    "t i a": "tia",
                    "j c c": "jcc",
                    "u s a": "usa",
                    "u k": "uk",
                    "t v": "tv",
                    "T V": "tv",
                    "p h d":"phd",
                    "ph.d": "phd",
                    "Ph.d": "phd",
                    "Ph.D": "phd",
                    "e e g": "eeg",
                    "e n t": "ent",
                    "i c u": "icu",
                    "ic u": "icu",
                    "head stands": "headstands",
                    "t-e-s-t": "t e s t",
                    "bull's eyes": "bullseyes",
                    'bulls eyes': "bullseyes",
                    "t e s t": "test",
                    "0 t": "o t",
       
    }

    if isinstance(text, str) == False:
        return text

    if pd.isna(text):
        print("NA exists for this file")

    text = remove_trailing_punctuations(text)

    # # use text replace to replace the words
    # for word, replacement in number_conversion_dict.items():
    #     text = re.sub(r"\b{}\b".format(re.escape(word)), replacement, text)

    split_words = text.split()
    # strip punctuation from each split word

    # for i in range(len(split_words)):
    #     if split_words[i].strip('.,!?;:"').lower() in pre_post_dict.keys():
    #         split_words[i] = pre_post_dict[split_words[i].lower()]
    split_words = [pre_post_dict.get(word.strip(
        '.,!?;:"').lower(), word) for word in split_words]
    text = ' '.join(split_words)

    # use text replace to replace the words
    for word, replacement in pre_post_dict2.items():
        text = re.sub(r"\b{}\b".format(re.escape(word)), replacement, text)
    return text

def modify_number_case(text):
    number_conversion_dict ={
                    "twenty twenties 728":"20 20s 728",
                    '50 8':"58", # fifty eight 
                    "70 four": "74",
                    "19/19/70? 4":"19 1974",
                    '50.8':"58",
                    "20. 123-4555":"21 234555",
                    "70.2":"72",
                    "2 a half":"2 5",
                    "2 and half":"2 5",
                    "2 and a half":"2 5",
                    "3 1/2":"3 5",
                    "3 a half":"3 5",
                    "20 3 or 24":"23 or 24",
                    "1990.5 early 1995":"1995 early 1995",
                    "seven seventeenth":"7 17th",
                    "go up to 100 and 40":"go up to 140",
                    "2000 and 3, 2008":"2003 2008",
                    "2000. And. 3":"2003",
                    "2000 and 2000 and three 2000":"2000 and 2003 2000",
                    "2000 67":"2006 7",
                    "2 1/2": "2 5"  # particularly for Azure transcription}
    }

    for word, replacement in number_conversion_dict.items():
        text = re.sub(r"\b{}\b".format(re.escape(word)), replacement, text)

    return text
    

def convert_time_to_words(text):
    '''
    if time is in the format of 1:00, 2:00, 3:00, 4:00, 5:00, 6:00, 7:00, 8:00, 9:00, 10:00,...
    convert it to "one o'clock", "two o'clock", "three o'clock", "four o'clock"...
    '''
    text = text.replace(":00", " ")
    text = num2words(int(text))
    text = text.replace(text, text+" o'clock ")

    return text


def number_to_word(text):
    """
    Convert number to words, accounting for special cases like date, time, dollar sign
    """
    if pd.isna(text):
        print("NA exists for this file")
        pass
    else:

        new_list = []
        date_suffix = ["th", "1st", "rd", "nd"]
        if "-" in text:
            # print("----------dash exist")
            text = text.replace("-", " ")
            # print(text)

        for token in text.split():
            if bool(re.search(r'\d', token)) == True:
                if token.isdigit():
                    # print(token,"is a digit")
                    if len(token) >= 4:  # detect if number is a year number
                        word = num2words(token, to='year')
                        # print("converted word is"+num2words(token,to='year'))
                    else:
                        word = num2words(token)
                        # print("converted word is",word)
                    new_list.append(word)
                elif "%" in token:  # detect if % exists in word as a specicial character and convert it to "percent"
                    print(" \%\ exists")
                    # remove punctuations for the word
                    word = re.sub(r'[^\w\s]|_', ' ', token)
                    word = num2words(word)  # convert it to word from number
                    new_list.append(word)
                    new_list.append("percent")

                elif ":" in token:  # detect if : exists in word
                    print(" \:\ exists")
                    # if token in ["1:00","2:00","3:00","4:00","5:00","6:00","7:00","8:00","9:00","10:00",
                    #             "11:00","12:00"]:
                    #     word = convert_time_to_words(token)
                    #     new_list.append(word)
                    # else:
                    word_list = token.split(":")
                    for word in word_list:
                        word = num2words((re.sub(r'[^\w\s\d]', '', word)))
                        new_list.append(word)

                elif re.match(r"\$[^\]]+", token, re.I):  # deal with $ to 'dollars'
                    money = num2words(token[1:])
                    if token[1:] in ["1"]:
                        new_word = money + " dollar"
                    else:
                        new_word = money + " dollars"
                    new_list.append(new_word)

                elif re.match(r"\£[^\]]+", token, re.I):
                    money = num2words(token[1:])
                    new_word = money + " pound"  # pound is not plural for currency
                    new_list.append(new_word)

                elif re.search(r'\d', token) and re.search('th|1st|nd|rd', token):
                    token = re.sub(r'[^\w\s\d]', '', token)
                    # print(token)
                    token = token.replace('th', "").replace(
                        'nd', "").replace('st', "").replace("rd", "")
                    # print(token)
                    word = num2words(token, to="ordinal")
                    new_list.append(word)
                else:

                    if len(token) >= 4 and token.isdigit():  # detect if number is a year number
                        word = num2words(token, to='year')
                    elif len(token) < 4 and token.isdigit():
                        word = num2words(token)
                    elif token.isdigit() == False:
                        token = re.sub(r'[^\w\s\d]', '', token)
                        # print(token)
                        non_digit_word = re.sub(r'[\d]', '', token)
                        # print(non_digit_word)
                        new_list.append(non_digit_word)
                        word = num2words(re.sub(r'[^\d]', '', token))
                        # print(word)
                    new_list.append(word)

            else:
                new_list.append(token)

        text = ' '.join(new_list)
       # print(text)
        return text


def removeSpace_betweenDigits(text):
    '''
    This function aims to remove any space between digits. The purpose of this is to avoid the difference induced by different translation techniques from different ASRs.
    '''

    if pd.isna(text):
        print("NA exists for this file")
        return text
    else:
        new_list = []
        for token in text.split():
            if bool(re.search(r'\d', token)):
                # remove punctuations for the word
                token = re.sub(r'[^\w\s]|_', '', token)
                # remove space between digits
                token = re.sub(r'(?<=\d) +(?=\d)', '', token)
                new_list.append(token)
            else:
                new_list.append(token)
        text = ' '.join(new_list)

    return text


def remove_spaces_between_numbers(text):
    # Use regular expression to find spaces between digits and numbers
    # and replace them with no space
    modified_text = re.sub(r'(?<=\b\d)[\s\W]+(?=\d\b)', '', text)
    return modified_text


# Main cleaning function

# def clean_text(text,is_groundtruth=False):
#     """
#     This function cleans the text in the following steps:
#     1. Remove diarization pattern
#     2. Standardize text that includes time "XX o'clock"
#     3. Fix spelling
#     4. Process text using Whisper normalizer
#     5. Remove filler words that are not in the Whisper normalizer
#     6. Remove all punctuations including currency sign without any whitespace and punctuations in time like 5.30 from whisper
#     7. For any token that are numeric, we insert a whitespace between digits
#     """

#     # # remove diarization pattern
#     pattern = r"Speaker \d+\s{2,}\d{2}:\d{2}:\d{2}"
#     text = re.sub(pattern, "", text)


#     # standardize text that includes time ":"
#     tokens = text.split()
#     for i in range(len(tokens)):
#         if tokens[i].endswith(":00") and tokens[i].split(":")[0].isdigit():
#             hour = int(tokens[i].split(":")[0])
#             tokens[i] = num2words(hour) + " o'clock"
#     text = " ".join(tokens)

#     # fix spelling 
#     text = spelling_rematch(text)

#     # process text using Whisper normalizer
#     text = english_normalizer(text)

#     # remove filler words
#     filler_words = ['hmhm', 'uhhuh', 'emmm', 'huh', 'umm', 'ugh', 'hm', 'uhuh', 'eh', 'uhh', 'mmhmm']
#     tokens  = [word.lower() for word in text.split() if re.sub('\,','',word.lower()) not in filler_words]
#     text= ' '.join(tokens) 

#     # remove all punctuations including currency sign without any whitespace 
#     text = re.sub(r'[^\w\s]|_', ' ',text) # this step occurs to remove punctuations in time like 5.30 from whisper

#     # for any token that are numeric, we insert a whitespace between digits 
#     tokens = text.split()

#     for i in range(len(tokens)):
#         if tokens[i] =="one": # standardized all ones at the end since the normalizer does not convert ones sometimes
#             tokens[i]=="1"
#         if re.match(r'^\d+$', tokens[i]):
#             # convert number to a string 
#             tokens[i] = str(tokens[i])
#             # insert a whitespace between digits
#             spaced_number = ' '.join(tokens[i])
#             tokens[i] = spaced_number

#     text = " ".join(tokens)
#     text = spelling_rematch(text)
#     # print("cleaned text: ",text)
#     return text

def wer_calc(transcripts, human_clean_col, asr_clean_col):
    # Calculate WER
    new_transcripts = transcripts.copy()
    ground_truth = transcripts[human_clean_col].tolist()
    for col in asr_clean_col:
        new_transcripts[col] = new_transcripts[col].replace(
            np.nan, '', regex=True)
        asr_trans = new_transcripts[col].tolist()
        wer_list = []
        for i in range(len(ground_truth)):
            # check if the ground truth is empty and append NA to the wer_list
            if ground_truth[i] == "" or ground_truth is None:
                print("ground truth is empty")
                wer_list.append("NA")
            else:
                wer_list.append(wer(ground_truth[i], asr_trans[i]))

        new_transcripts[human_clean_col+"_"+col+"_wer"] = wer_list

    return new_transcripts


def remove_fragments(text):
    """
    Remove fragments from the text. 
    """
    # remove fragments
    text_tokens = text.split()
    for i in range(len(text_tokens)-1):
        # check if the token is a fragment
        current_token = str(text_tokens[i])
        next_token = str(text_tokens[i+1]) if i+1 < len(text_tokens) else None
        # print(current_token,next_token)
        if len(current_token) < 3:
            if next_token.startswith(current_token) & (current_token != next_token):
                # remove the current token
                text_tokens[i] = ""
    text = " ".join(text_tokens)
    # remove any extra whitespace
    text = re.sub(' +', ' ', text)
    return text
