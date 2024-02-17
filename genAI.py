import requests
from openai import OpenAI
import textwrap
import os
from getpass import getpass


#### Prompt ####

RULE_PROMPT = '''
The criterions:

The Innocence Project ONLY considers post-conviction DNA cases where:
1. The trial has been completed, an appeal has taken place and the person is serving their
sentence.
2. There is physical evidence that, if subjected to DNA testing, will prove that the applicant is innocent. This means that physical evidence was collected – for example blood, bodily fluids, clothing, hair – and if that evidence can be found and tested, the test will prove that the applicant could not have committed the crime. The applicant must have been convicted of a crime. We do not review claims where the applicant was wrongfully suspected, arrested or charged, but not convicted.
3. The crime occurred in the United States, but not in: Arizona, California, Illinois, Michigan, or Ohio. Our intake is currently closed to these states. Please write to the local innocence organizations in those States for assistance.
The Innocence Project only reviews a limited number of cases where DNA testing cannot prove innocence. Further the Innocence Project does NOT handle the following types of cases:
1. Consent / Transaction: The applicant claims that there was no crime because the victim consented to the events (e.g. consensual intercourse) and/or received some form of payment in exchange for the activity (e.g. prostitution, drug transaction)
2. Self-defense / Justification: The applicant admits to causing the injury/death but believes the acts were justified because the applicant acted in self-defense or to protect other people.
3. Sustained Abuse: The applicant is accused of crime against the victim(s) that happened more than once and over a period of time (e.g., allegations of long-term sexual abuse of a family member).
4. Illegal Possession / Distribution of any controlled substance: The applicant is only challenging a possession or distribution charge.
5. RICO / Hobbs Act: The applicant has been charged under the Racketeer Influenced and Corrupt Organizations Act (RICO) or the Hobbs Act.
6. DWI or DUI: The applicant was convicted of driving while intoxicated (DWI) or driving under the influence (DUI).
7. Fraud / Identity Theft / Forgery: The applicant was only convicted of one or more of these crimes.
8. Stalking / Harassment: The primary charges against the applicant involve stalking and/or harassment.
9. Sentencing Reduction / Overcharged: The applicant wants to challenge the charge or length of sentence but does not claim innocence.

'''

INSTRUCTIONS = '''
Can you first check if there's any missing information in the entire text including Additional Information? If so, tell me all the places that are missing. If not, check if any criterion was violate in the additional information? If so, tell me which criterion was violated. If not, just tell me that there's no missing information and no criterion was violated.
'''

# Good Example
INPUT_PROMPT1 = '''
- Name of Applicant: Jane Smith
- Inmate Number: 987654321
- Current Address of Applicant: Women's Correctional Facility, 5678 Freedom Road, Liberty City, State, 101112
- Crime(s) Convicted Of: Second-degree murder
- Date(s) of the Crime(s): October 22, 2019
- Date of Arrest: November 1, 2019
- Date Crime Was Reported: October 23, 2019
- Date of Conviction: September 10, 2020
- City, County, and State of Conviction: Liberty City, Liberty County, AnyState (excluding AZ, CA, IL, MI, OH)
- Sentence Received: 25 years

Additional Information:
1. Basic Facts: Accused of being involved in a fatal altercation outside a nightclub. The incident took place on October 22, 2019, around 11:30 PM near Club Nightlife in Liberty City.
2. Where were you at the time of the crime(s)? Working a night shift at a local diner, corroborated by timecards and witness statements from colleagues.
3. Do you know the victim(s)? Only knew the victim casually through mutual acquaintances; did not have any direct involvement or conflict.
4. Are you claiming innocence of all the charges/convictions? Yes, completely innocent of the charges.

'''

# Violation Type: This example violates the "Consent/Transaction" criterion, as the applicant acknowledges the act but disputes its criminality based on consent and transaction.
INPUT_PROMPT2 = '''
- Name of Applicant: John Doe
- Inmate Number: 12345678
- Current Address of Applicant: State Penitentiary, City, State
- Crime(s) Convicted Of: Prostitution
- Date(s) of the Crime(s): June 15, 2020
- Date of Arrest: June 16, 2020
- Date Crime was Reported: June 15, 2020
- Date of Conviction: December 1, 2020
- City, County, and State of Conviction: Springfield, Greene County, State
- Sentence Received: 3 years in state prison

Additional Information:
1. Basic Facts: The applicant was accused of engaging in prostitution by accepting money in exchange for sexual services on the night of June 15, 2020, in downtown Springfield.
2. Where were you at the time of the crime(s)? The applicant claims to have been at the location of the crime, engaging in the transaction.
3. Do you know the victim(s)? The applicant acknowledges knowing the person who paid for the services but disputes the legality of the transaction.
4. Claiming Innocence? The applicant does not claim innocence of the actions but disputes the criminality based on mutual consent.

'''

# Violation Type: This example violates the "Self-defense/Justification" criterion, as the applicant admits to causing harm but claims it was justified as self-defense.
INPUT_PROMPT3 = '''
- Name of Applicant: Jane Smith
- Inmate Number: 87654321
- Current Address of Applicant: Women's Correctional Facility, City, State
- Crime(s) Convicted Of: Manslaughter
- Date(s) of the Crime(s): April 10, 2019
- Date of Arrest: April 12, 2019
- Date Crime was Reported: April 11, 2019
- Date of Conviction: November 5, 2019
- City, County, and State of Conviction: Lake City, Lake County, State
- Sentence Received: 10 years in state prison

Additional Information:
1. Basic Facts: The applicant was involved in a physical altercation that resulted in the death of another individual. The applicant claims the act was in self-defense after being attacked.
2. Where were you at the time of the crime(s)? The applicant was at the scene of the altercation, in a public park in Lake City.
3. Do you know the victim(s)? The applicant did not know the victim prior to the altercation.
4. Claiming Innocence? The applicant claims innocence based on self-defense, arguing that the actions were justified to protect herself from harm.
'''

# Missing Information: date of arrest and basic facts are missing
INPUT_PROMPT4 = '''
- Name of Applicant: Emily Johnson
- Inmate Number: 99887766
- Current Address of Applicant: Central Women's Facility, City, State
- Crime(s) Convicted Of: Burglary
- Date(s) of the Crime(s): August 23, 2018
- Date of Arrest: 
- Date Crime was Reported: August 23, 2018
- Date of Conviction: March 14, 2019
- City, County, and State of Conviction: Dover, Kent County, State
- Sentence Received: 5 years in state prison

Additional Information:
1. Basic Facts: 
2. Where were you at the time of the crime(s)? At home with family.
3. Do you know the victim(s)? No, I do not know the victim(s).
4. Claiming Innocence? Yes, I claim complete innocence of the charges.
'''

INPUT_PROMPTS = [INPUT_PROMPT1, INPUT_PROMPT2, INPUT_PROMPT3, INPUT_PROMPT4]

#### API Calling ####


## Llama2 API
# resp = requests.post(
#     "https://model-2qjd102q.api.baseten.co/production/predict",
#     headers={"Authorization": "Api-Key C0pAvn1v.F8wJ6muVWfA0WN0BzJYBoUJ1vHJI9QFx"},
#     json={'top_p': 0.75, 'prompt': PROMPT, 'num_beams': 4, 'temperature': 0.1, 'max_length': 4096},
# )

# print(resp.json())

## OpenAI API

if 'OPENAI_API_KEY' not in os.environ:
    os.environ['OPENAI_API_KEY'] = "TYPE YOUR OPENAI API KEY"
    client = OpenAI()

# def get_response(i=0):
#     response = client.chat.completions.create(
#         model="gpt-4",
#         messages=[
#             {
#                 "role": "system",
#                 "content": RULE_PROMPT
#             },
#             {
#                 "role": "user",
#                 "content": INPUT_PROMPTS[i]
#             },
#             {
#                 "role": "assistant",
#                 "content": INSTRUCTIONS
#             }
#         ],
#         temperature=1,
#         max_tokens=256,
#         top_p=1,
#         frequency_penalty=0,
#         presence_penalty=0
#     )
#     return wrap_text(response.choices[0].message.content)

def get_response(input_prompt):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": RULE_PROMPT
            },
            {
                "role": "user",
                "content": input_prompt
            },
            {
                "role": "assistant",
                "content": INSTRUCTIONS
            }
        ],
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return wrap_text(response.choices[0].message.content)

def wrap_text(text: str, width: int = 80) -> str:
    """
    Keeps text output narrow enough to easily be read
    """
    lines = text.split("\n")
    wrapped_lines = [textwrap.fill(line, width) for line in lines]
    return "\n".join(wrapped_lines)

# print(wrap_text(response.choices[0].message.content))