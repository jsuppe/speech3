#!/usr/bin/env python3
"""
Download the Archive.org "100 Best Speeches (USA)" collection.
70 audio files of famous American speeches.
"""

import os
import time
import requests
from pathlib import Path

AUDIO_DIR = Path("/home/melchior/speech3/audio/oratory")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# Archive.org 100 Best Speeches (USA) - with speaker names and titles
# Based on the collection at: https://archive.org/details/100-Best--Speeches
SPEECHES_100_BEST = [
    ("Anita Hill", "Sexual Harassment Hearings Testimony", "https://archive.org/download/100-Best--Speeches/AFH19911011_64kb.mp3"),
    ("Ann Richards", "1988 DNC Keynote", "https://archive.org/download/100-Best--Speeches/AR_19880719_64kb.mp3"),
    ("Barbara Bush", "Wellesley Commencement", "https://archive.org/download/100-Best--Speeches/BB_19900601_64kb.mp3"),
    ("Bill Clinton", "Oklahoma Bombing Memorial", "https://archive.org/download/100-Best--Speeches/BC_19950423_64kb.mp3"),
    ("Barry Goldwater", "1964 RNC Address", "https://archive.org/download/100-Best--Speeches/BG_19640716_64kb.mp3"),
    ("Barbara Jordan", "Statement on Impeachment", "https://archive.org/download/100-Best--Speeches/BJ_19740725_64kb.mp3"),
    ("Barbara Jordan", "1976 DNC Keynote", "https://archive.org/download/100-Best--Speeches/BJ_19760712_64kb.mp3"),
    ("Dwight Eisenhower", "Farewell Address", "https://archive.org/download/100-Best--Speeches/DDE19610117_64kb.mp3"),
    ("Dwight Eisenhower", "Atoms for Peace", "https://archive.org/download/100-Best--Speeches/DDE19631208_64kb.mp3"),
    ("Douglas MacArthur", "Farewell to Congress 1951", "https://archive.org/download/100-Best--Speeches/DM_19510419_64kb.mp3"),
    ("Douglas MacArthur", "Thayer Award Address", "https://archive.org/download/100-Best--Speeches/DM_19620512_64kb.mp3"),
    ("Elizabeth Glaser", "1992 DNC Address", "https://archive.org/download/100-Best--Speeches/EG_19920714_64kb.mp3"),
    ("Ted Kennedy", "Tribute to RFK", "https://archive.org/download/100-Best--Speeches/EK_19680608_64kb.mp3"),
    ("Ted Kennedy", "Chappaquiddick", "https://archive.org/download/100-Best--Speeches/EK_19690725_64kb.mp3"),
    ("Ted Kennedy", "1980 DNC Address", "https://archive.org/download/100-Best--Speeches/EK_19800812_64kb.mp3"),
    ("Ted Kennedy", "Truth and Tolerance", "https://archive.org/download/100-Best--Speeches/EK_19831003_64kb.mp3"),
    ("Eleanor Roosevelt", "UN Declaration of Human Rights", "https://archive.org/download/100-Best--Speeches/ER_19481209_64kb.mp3"),
    ("Elie Wiesel", "Perils of Indifference", "https://archive.org/download/100-Best--Speeches/EW_19990412_64kb.mp3"),
    ("FDR", "First Inaugural Address", "https://archive.org/download/100-Best--Speeches/FDR19330304_64kb.mp3"),
    ("FDR", "First Fireside Chat", "https://archive.org/download/100-Best--Speeches/FDR19330312_64kb.mp3"),
    ("FDR", "Arsenal of Democracy", "https://archive.org/download/100-Best--Speeches/FDR19401229_64kb.mp3"),
    ("FDR", "Four Freedoms", "https://archive.org/download/100-Best--Speeches/FDR19411206_64kb.mp3"),
    ("FDR", "Pearl Harbor Address", "https://archive.org/download/100-Best--Speeches/FDR19411208_64kb.mp3"),
    ("George Marshall", "The Marshall Plan", "https://archive.org/download/100-Best--Speeches/GCM19470605_64kb.mp3"),
    ("Geraldine Ferraro", "1984 VP Nomination", "https://archive.org/download/100-Best--Speeches/GF_19840719_64kb.mp3"),
    ("Gerald Ford", "Inaugural Address", "https://archive.org/download/100-Best--Speeches/GRF19740809_64kb.mp3"),
    ("Gerald Ford", "Nixon Pardon Address", "https://archive.org/download/100-Best--Speeches/GRF19740908_64kb.mp3"),
    ("Harry Truman", "Truman Doctrine", "https://archive.org/download/100-Best--Speeches/HST19470312_64kb.mp3"),
    ("Jimmy Carter", "Malaise Speech", "https://archive.org/download/100-Best--Speeches/JC_19790715_64kb.mp3"),
    ("JFK", "Houston Ministerial Association", "https://archive.org/download/100-Best--Speeches/JFK19600912_64kb.mp3"),
    ("JFK", "Inaugural Address", "https://archive.org/download/100-Best--Speeches/JFK19610120_64kb.mp3"),
    ("JFK", "Cuban Missile Crisis", "https://archive.org/download/100-Best--Speeches/JFK19621022_64kb.mp3"),
    ("JFK", "American University Commencement", "https://archive.org/download/100-Best--Speeches/JFK19630610_64kb.mp3"),
    ("JFK", "Civil Rights Address", "https://archive.org/download/100-Best--Speeches/JFK19630611_64kb.mp3"),
    ("JFK", "Ich bin ein Berliner", "https://archive.org/download/100-Best--Speeches/JFK19630626_64kb.mp3"),
    ("Jesse Jackson", "1984 DNC Part 1", "https://archive.org/download/100-Best--Speeches/JJ_19840718.1_64kb.mp3"),
    ("Jesse Jackson", "1984 DNC Part 2", "https://archive.org/download/100-Best--Speeches/JJ_19840718.2_64kb.mp3"),
    ("Jesse Jackson", "1984 DNC Part 3", "https://archive.org/download/100-Best--Speeches/JJ_19840718.3_64kb.mp3"),
    ("Jesse Jackson", "1988 DNC Part 1", "https://archive.org/download/100-Best--Speeches/JJ_19880719.1_64kb.mp3"),
    ("Jesse Jackson", "1988 DNC Part 2", "https://archive.org/download/100-Best--Speeches/JJ_19880719.2_64kb.mp3"),
    ("Joseph Welch", "McCarthy Welch Exchange", "https://archive.org/download/100-Best--Speeches/JNW19540609_64kb.mp3"),
    ("LBJ", "Let Us Continue", "https://archive.org/download/100-Best--Speeches/LBJ19631127_64kb.mp3"),
    ("LBJ", "Great Society", "https://archive.org/download/100-Best--Speeches/LBJ19640522_64kb.mp3"),
    ("LBJ", "We Shall Overcome", "https://archive.org/download/100-Best--Speeches/LBJ19650315_64kb.mp3"),
    ("LBJ", "Not Seeking Re-election", "https://archive.org/download/100-Best--Speeches/LBJ19680331_64kb.mp3"),
    ("Mario Cuomo", "1984 DNC Keynote", "https://archive.org/download/100-Best--Speeches/MC_19840716_64kb.mp3"),
    ("Mario Cuomo", "Religious Belief Public Morality", "https://archive.org/download/100-Best--Speeches/MC_19840913_64kb.mp3"),
    ("Mary Fisher", "Whisper of AIDS", "https://archive.org/download/100-Best--Speeches/MF_19920819_64kb.mp3"),
    ("MLK", "Declaration Against Vietnam War", "https://archive.org/download/100-Best--Speeches/MLK19670404_64kb.mp3"),
    ("MLK", "Ive Been to the Mountaintop", "https://archive.org/download/100-Best--Speeches/MLK19680403_64kb.mp3"),
    ("MLK", "I Have a Dream", "https://archive.org/download/100-Best--Speeches/MLK19830828_64kb.mp3"),
    ("Malcolm X", "Message to the Grassroots", "https://archive.org/download/100-Best--Speeches/MX_19631010_64kb.mp3"),
    ("Malcolm X", "Ballot or Bullet Part 1", "https://archive.org/download/100-Best--Speeches/MX_19640412.1_64kb.mp3"),
    ("Malcolm X", "Ballot or Bullet Part 2", "https://archive.org/download/100-Best--Speeches/MX_19640412.2_64kb.mp3"),
    ("Newton Minow", "Television Public Interest", "https://archive.org/download/100-Best--Speeches/NM_19610509_64kb.mp3"),
    ("RFK", "MLK Assassination Statement", "https://archive.org/download/100-Best--Speeches/RFK19680404_64kb.mp3"),
    ("Nixon", "Checkers Speech", "https://archive.org/download/100-Best--Speeches/RMN19520923_64kb.mp3"),
    ("Nixon", "Silent Majority", "https://archive.org/download/100-Best--Speeches/RMN19691103_64kb.mp3"),
    ("Nixon", "Cambodian Incursion", "https://archive.org/download/100-Best--Speeches/RMN19700430_64kb.mp3"),
    ("Nixon", "Resignation Address", "https://archive.org/download/100-Best--Speeches/RMN19740808_64kb.mp3"),
    ("Reagan", "A Time for Choosing", "https://archive.org/download/100-Best--Speeches/RR_19641027_64kb.mp3"),
    ("Reagan", "First Inaugural", "https://archive.org/download/100-Best--Speeches/RR_19810120_64kb.mp3"),
    ("Reagan", "Evil Empire", "https://archive.org/download/100-Best--Speeches/RR_19830308_64kb.mp3"),
    ("Reagan", "D-Day 40th Anniversary", "https://archive.org/download/100-Best--Speeches/RR_19840606_64kb.mp3"),
    ("Reagan", "Challenger Disaster", "https://archive.org/download/100-Best--Speeches/RR_19860128_64kb.mp3"),
    ("Reagan", "Tear Down This Wall", "https://archive.org/download/100-Best--Speeches/RR_19870612_64kb.mp3"),
    ("Spiro Agnew", "TV News Coverage", "https://archive.org/download/100-Best--Speeches/SA_19691113_64kb.mp3"),
    ("Stokely Carmichael", "Black Power", "https://archive.org/download/100-Best--Speeches/SC_19661000_64kb.mp3"),
    ("William Faulkner", "Nobel Prize Acceptance", "https://archive.org/download/100-Best--Speeches/WF_19501210_64kb.mp3"),
    ("William Jennings Bryan", "Cross of Gold", "https://archive.org/download/100-Best--Speeches/WJB19000808_64kb.mp3"),
]

def download_file(speaker, title, url):
    """Download a speech file."""
    safe_name = f"{speaker}_{title}".replace(" ", "_").replace("'", "").replace("/", "-")[:80]
    safe_name = "".join(c for c in safe_name if c.isalnum() or c in "_-")
    filename = AUDIO_DIR / f"{safe_name}.mp3"
    
    if filename.exists():
        print(f"  SKIP: {filename.name} (exists)")
        return True
    
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        filename.write_bytes(response.content)
        print(f"  ✓ Downloaded: {filename.name} ({len(response.content)//1024} KB)")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def main():
    print("=" * 60)
    print("100 BEST SPEECHES DOWNLOADER")
    print(f"Speeches: {len(SPEECHES_100_BEST)}")
    print("=" * 60)
    
    success = 0
    for i, (speaker, title, url) in enumerate(SPEECHES_100_BEST):
        print(f"\n[{i+1}/{len(SPEECHES_100_BEST)}] {speaker} - {title}")
        if download_file(speaker, title, url):
            success += 1
        time.sleep(0.3)
    
    print(f"\n✓ Downloaded {success}/{len(SPEECHES_100_BEST)} speeches")

if __name__ == "__main__":
    main()
