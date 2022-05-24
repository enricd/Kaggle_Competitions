I started with the approach from Juan Sensio in his yt channel: https://www.youtube.com/watch?v=iXvO9fCOAIg&t=4220s&ab_channel=sensio for learning purposes and trying to build on that something better if possible. Later I managed to get pretty good results with it and we merged in a Team for the competition. Finally we got into the second position.

## Lesson Learnt:
- The fact that all observation labels for every sample are single class, I think that is misleading the models: a single sample will really have multiple classes of plants and animals but as they only have one label, when we train the models we tell them that there is that species but there other aren't there, which is not true. Even so, cross entropy for mulit class is probably not that much effected.

- There are locations with more than 30 possible species there, even hundreds. Then, in the top 30 error prediction, there are little chances to guess the correct one even with the best model possible.

- Maintain a better experimental tracking and structure specially after more than 30 different experiements and models, small changes can break good past trained models.
