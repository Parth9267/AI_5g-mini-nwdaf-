# Classroom demonstration and viva guide

## Opening explanation

"My project implements selected concepts from a paper on the Network Data
Analytics Function, or NWDAF, in 5G. I use the authors' published testbed data to
analyze device movement and predict the next serving cell. I also simulate the
paper's subscription and notification workflow using Python callbacks."

## A 3–5 minute demonstration

1. **Overview:** show 652 source location records, four devices, and four cells.
   Explain that the route chart counts transitions within registration segments.
   Select a device and show its movement timeline.
2. **Model lab:** show the 67/30 chronological split and the baseline. Explain
   that training labels must also be known before the cutoff. Point out the
   Decision Tree's 53.33% and Gradient Boosting's 50.00% as actual measured scores.
3. **Event replay:** leave All devices selected. Click Subscribe, then Next event.
   The callback receives the record and may produce a forecast if history exists.
   Click Advance 10 to show resolved forecasts alongside wrong predictions.
4. **Unsubscribe:** record the notification count, unsubscribe, then click Next
   event. The publisher advances while the received count remains unchanged.
5. **About the paper:** show the mapping table and state the implementation scope.

Keep Model lab open for questions about evaluation. Replay accuracy over a few
events may differ from the final holdout accuracy. Only a complete, uninterrupted,
all-device subscribed replay reproduces the full 30-example holdout result.

## Likely viva questions

**What is NWDAF?**
A 5G network function that collects network data and supplies analytics. This
project implements a small analytics and prediction workflow inspired by it.

**What are UE, gNB and handover?**
UE is a user device; gNB is a 5G base station. Handover transfers a device's
connection between serving cells. We observe cell transitions in recorded data.

**Where does the dataset come from?**
The paper authors' GitHub repository, specifically the February 3–18, 2025 folder.
The downloader pins a commit and records SHA-256 checksums. The data originates
from the authors' simulated network testbed, not our own network measurements.

**What is the AI component?**
Supervised multiclass classification. The model learns which next cell follows
observed current/previous cells, device identity and time features.

**How are labels created?**
Within one device and registration segment, the next different observed cell is
the label. The first visit lacks previous-cell history; the final visit has no
known future destination. Such rows cannot be ordinary labeled examples.

**Why a baseline?**
A classifier should be compared with a simple rule. The baseline predicts the
most common destination from the current cell, calculated on training data only.

**What is data leakage, and how do you prevent it?**
Leakage makes future or test information available during training. Our features
use current/past observations; training targets must occur before the cutoff;
the encoder is fit only on training rows. We never use recorded future duration
as an input. The live replay cannot see the future label.

**Why is your accuracy below 80.65%?**
The paper uses a different setup and richer features. This prototype additionally
uses conservative registration filtering and a chronological test. The small test
set has only 30 examples. We do not claim to reproduce the original percentage.

**Why is Decision Tree better here?**
It scored one more correct prediction than Gradient Boosting on this holdout.
That small difference does not establish that it is generally a better model.
Gradient Boosting uses the paper's deep-tree settings without holdout tuning.

**Is the model score a reliable confidence value?**
It is an uncalibrated estimated class probability. With a small dataset and deep
trees it can be very high even for wrong predictions.

**What does unsubscribe demonstrate?**
The bus removes the callback. New events no longer reach the analytics consumer.
History is cleared because missed events make continuous tracking unreliable.

**Is this the full paper implemented?**
No. It implements selected analytics, ML, and event-processing concepts. It does
not deploy the 5G core, modify AMF/SMF, or implement 3GPP service interfaces.

**What is your contribution?**
The small independent Python implementation: source-data preparation, explicit
temporal evaluation, classifier/baseline comparison, local event-replay engine,
interactive dashboard and verification. Credit the paper authors for the original
framework and dataset. Follow your course's rules on disclosing AI assistance.
