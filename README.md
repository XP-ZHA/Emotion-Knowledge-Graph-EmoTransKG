# emotion-knowledge-graph
This paper presents EmoTransKG, an innovative Emotion Knowledge Graph (EKG) that establishes the transformations and connections between emotions across diverse open-textual events.
Different from existing EKGs that primarily focus on linking emotion keywords (e.g., joyful, sad, and angry) to related terms or assigning sentiment dimension ratings to emotion words by humans, EmoTransKG aims to represent the general knowledge involved in the transformation of emotions.
Specifically, in conversations, successive emotions expressed by a single speaker are considered the head and tail entities, with open-textual utterances (events) occurring between them representing the relation.
To analyze whether the emotion knowledge expressed by such relations is justified, we develop EmoTransNet, a transformer-based translational model that interprets each relation as an operation that transforms the subject emotion into the object emotion.
Notably, our designed EmoTransNet, as a pluggable, seamlessly combine with almost all conversational emotion recognition (CER) models for emotion retrofitting.
Experimental results from two CER datasets show that incorporating EmoTransNet with baseline models yields significant and consistent improvements, while the visualization of entities and relations clarifies their roles in emotion transformations.
These experiments confirm the quality and effectiveness of EmoTransKG.
