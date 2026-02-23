#!/usr/bin/env python3
"""
Expanded Oratory Sources - Aiming for 1000+ speeches
Archive.org Greatest Speeches of 20th Century + more collections
"""

# Archive.org Greatest Speeches of the 20th Century
# Base URL: https://archive.org/download/Greatest_Speeches_of_the_20th_Century/
ARCHIVE_ORG_GREATEST = [
    ("Edward VIII", "Abdication Address", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/AbdicationAddress.mp3", "royal_address"),
    ("Charles de Gaulle", "Address from France", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/AddressfromFrance.mp3", "political_wartime"),
    ("Richard Nixon", "Address on Vietnam War Protests", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/AddressonVietnamWarProtests.mp3", "political_address"),
    ("Douglas MacArthur", "Address to Congress 1951", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/AddresstoCongress-1951.mp3", "military_address"),
    ("Wernher von Braun", "Address to Congress 1958", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/AddresstoCongress-1958.mp3", "scientific_address"),
    ("Gerald Ford", "Address to Congress 1974", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/AddresstoCongress-1974.mp3", "political_address"),
    ("Barbara Jordan", "Address to the Democratic Convention", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/AddresstotheDemocraticConvention.mp3", "political_keynote"),
    ("Ronald Reagan", "Address to the Nation", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/AddresstotheNation.mp3", "political_address"),
    ("Ronald Reagan", "Address to the Nation on the Berlin Wall", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/AddresstotheNationontheBerlinWall.mp3", "political_foreign"),
    ("Winston Churchill", "Address to the Nation on the RAF", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/AddresstotheNationontheR.A.F.mp3", "political_wartime"),
    ("Eleanor Roosevelt", "Address to the Women of America", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/AddresstotheWomenofAmerica.mp3", "advocacy"),
    ("Abbie Hoffman", "Address to the Yippie Convention", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/AddresstotheYippieConvention.mp3", "protest"),
    ("Richard Nixon", "Checkers Speech", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/CheckersSpeech.mp3", "political_defense"),
    ("Apollo 8 Crew", "Christmas Greeting from Space", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/ChristmasGreetingfromSpace.mp3", "commemorative"),
    ("Al Gore", "Concession Stand", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/ConcessionStand.mp3", "political_concession"),
    ("JFK", "Cuban Missile Crisis", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/ConfrontationoverPresenceofRussianMisslesinCuba.mp3", "political_crisis"),
    ("FDR", "Declaration of War Against Japan", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/DeclarationofWarAgainstJapan.mp3", "political_wartime"),
    ("JFK", "Democratic Convention Acceptance 1960", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/DemocraticConventionAcceptanceSpeech.mp3", "political_acceptance"),
    ("Richard Nixon", "Election Eve Campaign Speech", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/ElectionEveCampaignSpeech.mp3", "political_campaign"),
    ("Dwight Eisenhower", "Farewell Address", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/FarewellAddress.mp3", "political_farewell"),
    ("Lou Gehrig", "Farewell to Baseball 1939", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/FarewelltoBaseball-1939.mp3", "sports_farewell"),
    ("Babe Ruth", "Farewell to Baseball 1947", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/FarewelltoBaseball-1947.mp3", "sports_farewell"),
    ("Winston Churchill", "First Radio Address as Prime Minister", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/FirstRadioAddressasPrimeMinister.mp3", "political_wartime"),
    ("Thomas Edison", "First Recorded Promotional Message", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/FirstRecordedPromotionalMessageontheEdisonPhonograph.mp3", "historical"),
    ("Shirley Temple", "Greetings to the Children of England", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/GreetingstotheChildrenofEngland.mp3", "commemorative"),
    ("FDR", "Inaugural Address 1933", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/InauguralAddress-1933.mp3", "political_inaugural"),
    ("FDR", "Inaugural Address 1937", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/InauguralAddress-1937.mp3", "political_inaugural"),
    ("Harry Truman", "Inaugural Address 1949", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/InauguralAddress-1949.mp3", "political_inaugural"),
    ("JFK", "Inaugural Address 1961", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/InauguralAddress-1961.mp3", "political_inaugural"),
    ("Richard Nixon", "Inaugural Address 1969", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/InauguralAddress-1969.mp3", "political_inaugural"),
    ("Ronald Reagan", "Inaugural Address 1981", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/InauguralAddress-1981.mp3", "political_inaugural"),
    ("Mario Cuomo", "Keynote Address Democratic Convention", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/KeynoteAddressforDemocraticConvention.mp3", "political_keynote"),
    ("Stokely Carmichael", "On Black Power", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/OnBlackPower.mp3", "civil_rights"),
    ("Neville Chamberlain", "Return from Munich Conference", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/OnHisReturnfromtheMunichConference.mp3", "political_foreign"),
    ("Richard Nixon", "On Releasing Watergate Tapes", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/OnReleasingtheWatergateTapes.mp3", "political_crisis"),
    ("LBJ", "On Signing Civil Rights Bill", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/OnSigningoftheCivilRightsBill.mp3", "civil_rights"),
    ("John Foster Dulles", "On Fall of Dien Bien Phu", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/OntheFallofDienBienPhu.mp3", "political_foreign"),
    ("Adlai Stevenson", "Presidential Campaign Address 1952", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/PresidentialCampaignAddress-1952.mp3", "political_campaign"),
    ("Barry Goldwater", "Presidential Campaign Address 1964", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/PresidentialCampaignAddress-1964.mp3", "political_campaign"),
    ("Hubert Humphrey", "Presidential Campaign Address", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/PresidentialCampaignAddress.mp3", "political_campaign"),
    ("JFK vs Nixon", "Presidential Debate 1960", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/PresidentialDebate.mp3", "debate"),
    ("Hubert Humphrey", "Press Conference on Convention Riots", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/PressConferenceontheRiotsattheDemocraticConvention.mp3", "political_address"),
    ("Richard Nixon", "Republican Convention Address", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/RepublicanConventionAddress.mp3", "political_acceptance"),
    ("Spiro Agnew", "Resignation Address 1973", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/ResignationAddress-1973.mp3", "political_farewell"),
    ("Richard Nixon", "Resignation Address 1974", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/ResignationAddress-1974.mp3", "political_farewell"),
    ("Oliver North", "Iran-Contra Hearings Testimony", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/TestimonyattheIran-ContraHearings.mp3", "testimony"),
    ("Adlai Stevenson", "Cuban Missile Crisis UN", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/TheCubanMissileCrisis.mp3", "political_foreign"),
    ("Gerald Ford", "End of Vietnam War", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/TheEndoftheVietnamWar.mp3", "political_address"),
    ("Ezra Taft Benson", "Farmer and Republican Party", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/TheFarmerandtheRepublicanParty.mp3", "political_address"),
    ("John Glenn", "First American in Earth Orbit", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/TheFirstAmericaninEarthOrbit.mp3", "scientific_address"),
    ("Harry Truman", "First Atomic Bomb Attack on Japan", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/TheFirstAtomicBombAttackonJapan.mp3", "political_announcement"),
    ("Amelia Earhart", "Future of Women in Flying", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/TheFutureofWomeninFlying.mp3", "advocacy"),
    ("Winston Churchill", "The German Peril", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/TheGermanPeril.mp3", "political_wartime"),
    ("William Jennings Bryan", "The Ideal Republic", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/TheIdealRepublic.mp3", "political_campaign"),
    ("Nixon vs Khrushchev", "The Kitchen Debate", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/TheKitchenDebate.mp3", "debate"),
    ("Apollo 11", "The Moon Landing", "https://archive.org/download/Greatest_Speeches_of_the_20th_Century/TheMoonLanding.mp3", "historical"),
]

# Archive.org Famous Speeches collection  
ARCHIVE_ORG_FAMOUS = [
    ("FDR", "A Date Which Will Live in Infamy", "https://archive.org/download/fdr_a_date_which_will_live_in_infamy/fdr_a_date_which_will_live_in_infamy.mp3", "political_wartime"),
    ("Winston Churchill", "We Shall Fight on the Beaches", "https://archive.org/download/winstoncurchillweshallfightonthebeaches/winstoncurchillweshallfightonthebeaches.mp3", "political_wartime"),
    ("Winston Churchill", "Their Finest Hour", "https://archive.org/download/TheirFinestHour/TheirFinestHour.mp3", "political_wartime"),
]

# LibriVox Speeches and Orations (Public Domain audiobooks of famous speeches)
LIBRIVOX_SPEECHES = [
    ("Abraham Lincoln", "Gettysburg Address", "https://ia800206.us.archive.org/27/items/gettysburg_address_librivox/gettysburg_address.mp3", "political_commemorative"),
    ("Patrick Henry", "Give Me Liberty or Give Me Death", "https://ia802607.us.archive.org/28/items/give_me_liberty_or_give_me_death_librivox/give_me_liberty.mp3", "political_revolutionary"),
]

# More Archive.org speech collections
ARCHIVE_ORG_CIVIL_RIGHTS = [
    ("MLK", "Letter from Birmingham Jail Reading", "https://archive.org/download/MLKLetterFromBirminghamJail/MLK%20Letter%20From%20Birmingham%20Jail.mp3", "civil_rights"),
]

# TED-style talks and lectures - need to find public domain ones
# Note: TED talks themselves are copyrighted

def get_all_sources():
    """Combine all source lists."""
    all_speeches = []
    all_speeches.extend(ARCHIVE_ORG_GREATEST)
    all_speeches.extend(ARCHIVE_ORG_FAMOUS)
    all_speeches.extend(LIBRIVOX_SPEECHES)
    all_speeches.extend(ARCHIVE_ORG_CIVIL_RIGHTS)
    return all_speeches

if __name__ == "__main__":
    speeches = get_all_sources()
    print(f"Total expanded sources: {len(speeches)}")
    for s in speeches[:5]:
        print(f"  - {s[0]}: {s[1]}")
