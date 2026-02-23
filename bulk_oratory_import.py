#!/usr/bin/env python3
"""
Bulk Oratory Speech Importer
Downloads and analyzes speeches from American Rhetoric and other sources.
Target: 1000 oratory speeches
"""

import os
import sys
import json
import time
import sqlite3
import hashlib
import requests
import subprocess
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Paths
AUDIO_DIR = Path("/home/melchior/speech3/audio/oratory")
DB_PATH = "/home/melchior/speech3/speechscore.db"
STATE_FILE = Path("/home/melchior/speech3/oratory_import_state.json")
VENV_PYTHON = "/home/melchior/speech3/venv/bin/python"

# American Rhetoric Top 100 speeches with audio
AMERICAN_RHETORIC_SPEECHES = [
    # Format: (speaker, title, mp3_url, category)
    ("Martin Luther King Jr.", "I Have A Dream", "https://www.archive.org/download/MLKDream/MLKDream_64kb.mp3", "civil_rights"),
    ("John F. Kennedy", "Inaugural Address", "https://americanrhetoric.com/mp3clips/politicalspeeches/jfkinaugural2.mp3", "political_inaugural"),
    ("Franklin D. Roosevelt", "First Inaugural Address", "https://americanrhetoric.com/mp3clips/politicalspeeches/fdrfirstinaugural11223.mp3", "political_inaugural"),
    ("Franklin D. Roosevelt", "Pearl Harbor Address", "https://americanrhetoric.com/mp3clips/politicalspeeches/fdrwarmessage344.mp3", "political_wartime"),
    ("Barbara Jordan", "1976 DNC Keynote", "https://americanrhetoric.com/mp3clips/politicalspeeches/barbarajordan1976dnc.mp3", "political_keynote"),
    ("Richard Nixon", "Checkers Speech", "https://americanrhetoric.com/mp3clips/politicalspeeches/richardnixoncheckersmillercenter.mp3", "political_defense"),
    ("Malcolm X", "The Ballot or the Bullet Pt1", "https://archive.org/download/100-Best--Speeches/MX_19640412.1_64kb.mp3", "civil_rights"),
    ("Malcolm X", "The Ballot or the Bullet Pt2", "https://archive.org/download/100-Best--Speeches/MX_19640412.2_64kb.mp3", "civil_rights"),
    ("Ronald Reagan", "Challenger Disaster Address", "https://americanrhetoric.com/mp3clips/politicalspeeches/ronaldreaganchallengeraddress.mp3", "political_crisis"),
    ("John F. Kennedy", "Houston Ministerial Association", "https://americanrhetoric.com/mp3clips/politicalspeeches/jfkhoustonministers99999999999999999.mp3", "political_religious"),
    ("Lyndon B. Johnson", "We Shall Overcome", "https://americanrhetoric.com/mp3clips/politicalspeeches/lbjweshallovercome2233.mp3", "civil_rights"),
    ("Mario Cuomo", "1984 DNC Keynote", "https://ia601400.us.archive.org/21/items/100-Best--Speeches/MC_19840716_64kb.mp3", "political_keynote"),
    ("Jesse Jackson", "1984 DNC Address", "https://americanrhetoric.com/mp3clips/politicalspeeches/jessejackson1984dnc.mp3", "political_address"),
    ("Douglas MacArthur", "Farewell to Congress", "https://americanrhetoric.com/mp3clips/politicalspeeches/douglasmacarthurfarewell34323.mp3", "political_farewell"),
    ("Martin Luther King Jr.", "I've Been to the Mountaintop", "https://ia800909.us.archive.org/3/items/IHaveBeenToTheMountaintopFullSpeech/I%20Have%20Been%20to%20the%20Mountaintop%20Full%20Speech.mp3", "civil_rights"),
    ("Robert F. Kennedy", "On MLK Assassination", "https://americanrhetoric.com/mp3clips/politicalspeeches/rfkonmlkdeath45454.mp3", "political_eulogy"),
    ("Dwight Eisenhower", "Farewell Address", "https://americanrhetoric.com/mp3clips/politicalspeeches/dwighteisenhowerfarewell.mp3", "political_farewell"),
    ("Douglas MacArthur", "Duty Honor Country", "https://americanrhetoric.com/mp3clips/speeches/douglasmacarthurthayeraward.mp3", "military_address"),
    ("Richard Nixon", "Silent Majority", "https://americanrhetoric.com/mp3clips/politicalspeeches/richardnixonsilentmajority1122.mp3", "political_address"),
    ("John F. Kennedy", "Ich bin ein Berliner", "https://americanrhetoric.com/mp3clipsXE/politicalspeeches/jfkberlinspeechARXE.mp3", "political_foreign"),
    ("Ronald Reagan", "A Time for Choosing", "https://americanrhetoric.com/mp3clips/politicalspeeches/ronaldreaganatimeforchoosing.mp3", "political_campaign"),
    ("Franklin D. Roosevelt", "Arsenal of Democracy", "https://americanrhetoric.com/mp3clips/politicalspeeches/fdrarsenalofdemocracy7.mp3", "political_wartime"),
    ("Ronald Reagan", "Evil Empire", "https://americanrhetoric.com/mp3clips/politicalspeeches/ronaldreaganevilempire439.mp3", "political_foreign"),
    ("Ronald Reagan", "First Inaugural", "https://americanrhetoric.com/mp3clips/politicalspeeches/ronaldreaganfirstinaugural1122.mp3", "political_inaugural"),
    ("Franklin D. Roosevelt", "First Fireside Chat", "https://americanrhetoric.com/mp3clips/politicalspeeches/fdrfirstfiresidechat63496436943.mp3", "political_address"),
    ("Harry Truman", "Truman Doctrine", "https://americanrhetoric.com/mp3clips/politicalspeeches/harrytrumandoctrine.mp3", "political_foreign"),
    ("William Faulkner", "Nobel Prize Acceptance", "https://americanrhetoric.com/mp3clips/speeches/williamfaulknernobelprizeoriginal43438rew8.mp3", "literary_address"),
    ("Hillary Clinton", "Women's Rights Human Rights", "https://americanrhetoric.com/mp3clipsXE/politicalspeeches/hillaryclinton4thworldcongressARXE.mp3", "political_advocacy"),
    ("Dwight Eisenhower", "Atoms for Peace", "https://americanrhetoric.com/mp3clips/politicalspeeches/dwighteisenhoweratomsforpeace2233.mp3", "political_foreign"),
    ("John F. Kennedy", "American University Commencement", "https://americanrhetoric.com/mp3clips/politicalspeeches/jfkamericanuniversity4654.mp3", "commencement"),
    ("Ann Richards", "1988 DNC Keynote", "https://americanrhetoric.com/mp3clips/politicalspeeches/annrichards1988dnc.mp3", "political_keynote"),
    ("Richard Nixon", "Resignation Speech", "https://americanrhetoric.com/mp3clips/politicalspeeches/rnixonresignationspeechtwt9gf9.mp3", "political_farewell"),
    ("Franklin D. Roosevelt", "Four Freedoms", "https://americanrhetoric.com/mp3clips/politicalspeeches/fdrfourfreedoms.mp3", "political_address"),
    ("Martin Luther King Jr.", "A Time to Break Silence", "https://archive.org/download/BeyondVietnamATimeToBreakSilence4467/Beyond%20Vietnam-A%20Time%20to%20Break%20Silence%204-4-67.mp3", "civil_rights"),
    ("Barbara Bush", "Wellesley Commencement", "https://americanrhetoric.com/mp3clipsXE/politicalspeeches/barbarabushwellesleyARXE.mp3", "commencement"),
    ("John F. Kennedy", "Civil Rights Address", "https://americanrhetoric.com/mp3clips/politicalspeeches/jfkcivilrights.mp3", "civil_rights"),
    ("John F. Kennedy", "Cuban Missile Crisis", "https://americanrhetoric.com/mp3clips/politicalspeeches/jfkcubanmissilecrisis.mp3", "political_crisis"),
    ("Spiro Agnew", "Television News Coverage", "https://americanrhetoric.com/mp3clips/politicalspeeches/spiroagnewtvnewsgo2.mp3", "political_media"),
    ("Jesse Jackson", "1988 DNC Address Pt1", "https://archive.org/download/100-Best--Speeches/JJ_19880719.1_vbr.mp3", "political_address"),
    ("Jesse Jackson", "1988 DNC Address Pt2", "https://archive.org/download/100-Best--Speeches/JJ_19880719.2_vbr.mp3", "political_address"),
    ("Mary Fisher", "A Whisper of AIDS", "https://americanrhetoric.com/mp3clips/politicalspeeches/maryfisher1992rnc.mp3", "advocacy"),
    ("Lyndon B. Johnson", "The Great Society", "https://americanrhetoric.com/mp3clips/politicalspeeches/lbjthegreatsociety454529.mp3", "political_vision"),
    ("George Marshall", "The Marshall Plan", "https://americanrhetoric.com/mp3clips/politicalspeeches/georgemarshallplan1111.mp3", "political_foreign"),
    ("Ted Kennedy", "Truth and Tolerance", "https://americanrhetoric.com/mp3clips/politicalspeeches/tedkennedytruthandtolerance112.mp3", "political_religious"),
    ("Geraldine Ferraro", "VP Nomination Acceptance", "https://americanrhetoric.com/mp3clips/politicalspeeches/gferrarovicepresidentialacceptanceaddress.mp3", "political_acceptance"),
    ("Ronald Reagan", "D-Day 40th Anniversary", "https://americanrhetoric.com/mp3clips/politicalspeeches/ronaldrreagandday4534592.mp3", "commemorative"),
    ("Mario Cuomo", "Religious Belief and Public Morality", "https://americanrhetoric.com/mp3clips/politicalspeeches/mariocuomoreligiousbelief3434.mp3", "political_religious"),
    ("Ted Kennedy", "Chappaquiddick", "https://americanrhetoric.com/mp3clips/politicalspeeches/tedkennedychappaquiddickspeech.mp3", "political_personal"),
    ("Barry Goldwater", "1964 RNC Acceptance", "https://americanrhetoric.com/mp3clips/politicalspeeches/barrygoldwater1964rnc.mp3", "political_acceptance"),
    ("Stokely Carmichael", "Black Power", "https://download.publicradio.org/podcast/sound_learning/2009/02/13/stokelycarmichael_20090213_64.mp3", "civil_rights"),
    ("Newton Minow", "Television and Public Interest", "https://americanrhetoric.com/mp3clips/speeches/newtonminowtvandpublicinterest.mp3", "regulatory_address"),
    ("Ted Kennedy", "Eulogy for RFK", "https://americanrhetoric.com/mp3clips/speeches/tedkennedyonrfk45565656564.mp3", "eulogy"),
    ("Anita Hill", "Senate Judiciary Statement", "https://americanrhetoric.com/mp3clips/speeches/anitahillopeningstmtclarencethomashearings.mp3", "testimony"),
    ("Richard Nixon", "Cambodian Incursion", "https://americanrhetoric.com/mp3clips/politicalspeeches/rnixoncambodianincursion.mp3", "political_wartime"),
    ("Ted Kennedy", "1980 DNC Address", "https://americanrhetoric.com/mp3clips/politicalspeeches/tedkennedy1980dnc1122.mp3", "political_address"),
    ("Lyndon B. Johnson", "Not Seeking Re-Election", "https://americanrhetoric.com/mp3clips/politicalspeeches/lbjvietnamandreelection1122.mp3", "political_farewell"),
    ("Mario Savio", "Sproul Hall Sit-in", "https://americanrhetoric.com/mp3clips/politicalspeeches/mariosaviosproulhallsitin1122.mp3", "protest"),
    ("Elizabeth Glaser", "1992 DNC Address", "https://americanrhetoric.com/mp3clips/politicalspeeches/elizabethglaser1992dnc5343.mp3", "advocacy"),
    ("Gerald Ford", "Taking Oath of Office", "https://americanrhetoric.com/mp3clips/politicalspeeches/geraldfordoathofoffice.mp3", "political_inaugural"),
]

# Archive.org Greatest Speeches of the 20th Century
ARCHIVE_ORG_GREATEST = [
    ("Edward VIII", "Abdication Address", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/AbdicationAddress.mp3", "royal_address"),
    ("Charles de Gaulle", "Address from France", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/AddressfromFrance.mp3", "political_wartime"),
    ("Richard Nixon", "Address on Vietnam War Protests", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/AddressonVietnamWarProtests.mp3", "political_address"),
    ("Douglas MacArthur", "Address to Congress 1951", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/AddresstoCongress-1951.mp3", "military_address"),
    ("Wernher von Braun", "Address to Congress 1958", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/AddresstoCongress-1958.mp3", "scientific_address"),
    ("Gerald Ford", "Address to Congress 1974", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/AddresstoCongress-1974.mp3", "political_address"),
    ("Barbara Jordan", "DNC Address Archive", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/AddresstotheDemocraticConvention.mp3", "political_keynote"),
    ("Ronald Reagan", "Address to the Nation Archive", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/AddresstotheNation.mp3", "political_address"),
    ("Ronald Reagan", "Berlin Wall Address", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/AddresstotheNationontheBerlinWall.mp3", "political_foreign"),
    ("Winston Churchill", "RAF Address", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/AddresstotheNationontheR.A.F.mp3", "political_wartime"),
    ("Eleanor Roosevelt", "Address to Women of America", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/AddresstotheWomenofAmerica.mp3", "advocacy"),
    ("Abbie Hoffman", "Yippie Convention Address", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/AddresstotheYippieConvention.mp3", "protest"),
    ("Richard Nixon", "Checkers Speech Archive", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/CheckersSpeech.mp3", "political_defense"),
    ("Apollo 8 Crew", "Christmas Greeting from Space", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/ChristmasGreetingfromSpace.mp3", "commemorative"),
    ("Al Gore", "2000 Concession Speech", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/ConcessionStand.mp3", "political_concession"),
    ("JFK", "Cuban Missile Crisis Archive", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/ConfrontationoverPresenceofRussianMisslesinCuba.mp3", "political_crisis"),
    ("FDR", "Pearl Harbor Declaration", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/DeclarationofWarAgainstJapan.mp3", "political_wartime"),
    ("JFK", "1960 DNC Acceptance", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/DemocraticConventionAcceptanceSpeech.mp3", "political_acceptance"),
    ("Richard Nixon", "Election Eve 1968", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/ElectionEveCampaignSpeech.mp3", "political_campaign"),
    ("Dwight Eisenhower", "Farewell Address Archive", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/FarewellAddress.mp3", "political_farewell"),
    ("Lou Gehrig", "Luckiest Man Speech", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/FarewelltoBaseball-1939.mp3", "sports_farewell"),
    ("Babe Ruth", "Farewell Speech", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/FarewelltoBaseball-1947.mp3", "sports_farewell"),
    ("Winston Churchill", "Blood Toil Tears Sweat", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/FirstRadioAddressasPrimeMinister.mp3", "political_wartime"),
    ("Thomas Edison", "Phonograph Promotional", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/FirstRecordedPromotionalMessageontheEdisonPhonograph.mp3", "historical"),
    ("Shirley Temple", "Greetings to England", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/GreetingstotheChildrenofEngland.mp3", "commemorative"),
    ("FDR", "1933 Inaugural Archive", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/InauguralAddress-1933.mp3", "political_inaugural"),
    ("FDR", "1937 Inaugural", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/InauguralAddress-1937.mp3", "political_inaugural"),
    ("Harry Truman", "1949 Inaugural", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/InauguralAddress-1949.mp3", "political_inaugural"),
    ("JFK", "1961 Inaugural Archive", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/InauguralAddress-1961.mp3", "political_inaugural"),
    ("Richard Nixon", "1969 Inaugural", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/InauguralAddress-1969.mp3", "political_inaugural"),
    ("Ronald Reagan", "1981 Inaugural", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/InauguralAddress-1981.mp3", "political_inaugural"),
    ("Mario Cuomo", "DNC Keynote Archive", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/KeynoteAddressforDemocraticConvention.mp3", "political_keynote"),
    ("Stokely Carmichael", "Black Power Archive", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/OnBlackPower.mp3", "civil_rights"),
    ("Neville Chamberlain", "Peace in Our Time", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/OnHisReturnfromtheMunichConference.mp3", "political_foreign"),
    ("Richard Nixon", "Watergate Tapes Release", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/OnReleasingtheWatergateTapes.mp3", "political_crisis"),
    ("LBJ", "Civil Rights Bill Signing", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/OnSigningoftheCivilRightsBill.mp3", "civil_rights"),
    ("John Foster Dulles", "Dien Bien Phu Address", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/OntheFallofDienBienPhu.mp3", "political_foreign"),
    ("Adlai Stevenson", "1952 Campaign Address", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/PresidentialCampaignAddress-1952.mp3", "political_campaign"),
    ("Barry Goldwater", "1964 Campaign Address", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/PresidentialCampaignAddress-1964.mp3", "political_campaign"),
    ("Hubert Humphrey", "Presidential Campaign", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/PresidentialCampaignAddress.mp3", "political_campaign"),
    ("JFK vs Nixon", "1960 Presidential Debate", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/PresidentialDebate.mp3", "debate"),
    ("Hubert Humphrey", "Convention Riots Press", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/PressConferenceontheRiotsattheDemocraticConvention.mp3", "political_address"),
    ("Richard Nixon", "1968 RNC Address", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/RepublicanConventionAddress.mp3", "political_acceptance"),
    ("Spiro Agnew", "Resignation 1973", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/ResignationAddress-1973.mp3", "political_farewell"),
    ("Richard Nixon", "Resignation 1974", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/ResignationAddress-1974.mp3", "political_farewell"),
    ("Oliver North", "Iran-Contra Testimony", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/TestimonyattheIran-ContraHearings.mp3", "testimony"),
    ("Adlai Stevenson", "UN Cuban Missile Speech", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/TheCubanMissileCrisis.mp3", "political_foreign"),
    ("Gerald Ford", "Vietnam War End", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/TheEndoftheVietnamWar.mp3", "political_address"),
    ("Ezra Taft Benson", "Farmer and GOP", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/TheFarmerandtheRepublicanParty.mp3", "political_address"),
    ("John Glenn", "First Orbit Speech", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/TheFirstAmericaninEarthOrbit.mp3", "scientific_address"),
    ("Harry Truman", "Atomic Bomb Announcement", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/TheFirstAtomicBombAttackonJapan.mp3", "political_announcement"),
    ("Amelia Earhart", "Women in Aviation", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/TheFutureofWomeninFlying.mp3", "advocacy"),
    ("Winston Churchill", "German Peril Speech", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/TheGermanPeril.mp3", "political_wartime"),
    ("William Jennings Bryan", "Ideal Republic", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/TheIdealRepublic.mp3", "political_campaign"),
    ("Nixon vs Khrushchev", "Kitchen Debate", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/TheKitchenDebate.mp3", "debate"),
    ("Apollo 11", "Moon Landing Broadcast", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/TheMoonLanding.mp3", "historical"),
]

def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"downloaded": [], "processed": [], "failed": [], "stats": {"total": 0, "success": 0, "errors": 0}}

def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))

def download_speech(speaker, title, url, category):
    """Download a speech mp3 file."""
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create safe filename
    safe_name = f"{speaker}_{title}".replace(" ", "_").replace("/", "-").replace("'", "")[:80]
    safe_name = "".join(c for c in safe_name if c.isalnum() or c in "_-")
    filename = AUDIO_DIR / f"{safe_name}.mp3"
    
    if filename.exists():
        print(f"  Already exists: {filename.name}")
        return str(filename), True
    
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=60)
        response.raise_for_status()
        
        filename.write_bytes(response.content)
        print(f"  Downloaded: {filename.name} ({len(response.content) // 1024} KB)")
        return str(filename), True
    except Exception as e:
        print(f"  Error downloading {title}: {e}")
        return None, False

def analyze_speech(audio_path, speaker, title, category):
    """Run the speech analysis pipeline using the API endpoint."""
    try:
        import requests
        
        # Use the local API to analyze the speech
        with open(audio_path, 'rb') as f:
            files = {'file': (os.path.basename(audio_path), f, 'audio/mpeg')}
            data = {
                'title': title,
                'speaker': speaker,
                'category': category,
                'product': 'SpeakFit',
                'profile': 'default'
            }
            headers = {'X-API-Key': 'sk-5b7fd66025ead6b8731ef73b2c970f26'}
            
            response = requests.post(
                'http://localhost:8000/v1/analyze',
                files=files,
                data=data,
                headers=headers,
                timeout=300
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"  Analyzed: {title} (id={result.get('id', 'unknown')})")
                return True
            else:
                print(f"  API error for {title}: {response.status_code} - {response.text[:100]}")
                return False
    except Exception as e:
        print(f"  Analysis exception for {title}: {e}")
        return False

def main():
    print("=" * 60)
    print("ORATORY SPEECH BULK IMPORTER")
    print(f"Target: 1000 speeches | Started: {datetime.now()}")
    print("=" * 60)
    
    state = load_state()
    downloaded_urls = set(state["downloaded"])
    
    # Combine all speech sources
    all_speeches = []
    all_speeches.extend([("AmRhet", s) for s in AMERICAN_RHETORIC_SPEECHES])
    all_speeches.extend([("Archive", s) for s in ARCHIVE_ORG_GREATEST])
    
    total_speeches = len(all_speeches)
    print(f"\nTotal sources: {total_speeches} speeches")
    
    for i, (source, speech) in enumerate(all_speeches):
        speaker, title, url, category = speech
        
        if url in downloaded_urls:
            print(f"[{i+1}/{total_speeches}] SKIP (already processed): {speaker} - {title}")
            continue
            
        print(f"\n[{i+1}/{total_speeches}] [{source}] {speaker} - {title}")
        
        # Download
        audio_path, success = download_speech(speaker, title, url, category)
        if success and audio_path:
            state["downloaded"].append(url)
            state["stats"]["total"] += 1
            
            # Analyze
            if analyze_speech(audio_path, speaker, title, category):
                state["processed"].append(url)
                state["stats"]["success"] += 1
            else:
                state["stats"]["errors"] += 1
        else:
            state["failed"].append(url)
            state["stats"]["errors"] += 1
        
        save_state(state)
        time.sleep(0.3)  # Be nice to servers
    
    # Print summary
    print("\n" + "=" * 60)
    print("IMPORT SUMMARY")
    print("=" * 60)
    print(f"Total attempted: {state['stats']['total']}")
    print(f"Successfully processed: {state['stats']['success']}")
    print(f"Errors: {state['stats']['errors']}")
    print(f"State saved to: {STATE_FILE}")

if __name__ == "__main__":
    main()
