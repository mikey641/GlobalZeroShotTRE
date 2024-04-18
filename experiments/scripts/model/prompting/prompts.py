def task_description_v1():
    # this example is generated from the 139d3_temporal_Alon.json file
    desc = """
    Task Overview:
    You will be given a text with event mentions highlighted within it (identified by '<' and '>' symbols),
    Each event mention in the text is coupled with its mention ID immediately after (in parentheses). 
    Following the text, you will be given a list of event pairs in the format of event1(id1)/event2(id2), event2(id2)/event3(id3), ... 
    Your task is to determine the temporal relationships between all given pairs based on the provided text.

    Instructions:
    Consider only the starting time of events to determine the temporal relationships between them.

    Event Relationships:
    For each pair of events, you will assign one of the following temporal relationships based on the starting times of the events:

    Before: Event A started before Event B. For example, given the text "A traveler is <kidnapped>, and the police officers <said> the kidnapper is demanding money,". Event 'kidnapped' started before 'said', therefore the pair 'kidnapped'/'said' should equal 'before'.
    After: Event A started before Event B. Using the same example, for the pair 'said'/'kidnapped', you should now put the relation 'after', as 'said' happened after 'kidnapped'.
    Equal: Event A and B started at the same time. For example, given the text "They <filed> objections to the court, <claiming> that the suspects were treated unfairly". both 'filed' and 'claiming' are refering to the same thing and therefore happened at the same time, the relation between 'filed'/'claiming' or 'claiming'/'filed' should be 'equal'.
    Uncertain: The order of events cannot be determined from the context. For example, given the text "I <ate> a burger and <drank> a bottle of water for lunch today,". We cannot ascertain from the text whether 'ate' is earlier or later than 'drank', so the relation between 'ate'/'drank' should be 'uncertain'.

    To accurately assign the correct temporal relationships between events, look for explicit temporal indicators in the text. 
    These indicators may include specific dates, times, and keywords such as 'before', 'after', 'following, 'at the same time', etc. 

    Output Format: 
    The same list of events as given in the input with the resolved temporal relationships between them -- e.g., event1(id1)/event2(id2)=relation1,event2(id2)/event3(id3)=relation2, ... 

    Example:
    Consider this sample text with marked events:
    – " We 've just been Banksy'ed . " So declared Sotheby 's of London on Friday after a bizarre <stunt(12)> apparently pulled off by the elusive artist himself — one of his prints <shredded(55)> itself just after being <sold(37)> at auction for $ 1.4 million . Banksy has since <revealed(23)> how the <shredding(7)> of his 2006 " Girl With Balloon " took place , but some think the artist may have inadvertently revealed more than that : perhaps an image of himself , or at the very least a close assistant . Details and developments : The ' how ' : Banksy posted a video online showing that he <built(30)> a remote - controlled shredder into the print 's frame " a few years ago " so he could destroy the work if it ever went up for auction . The video also captures the initial moments of that happening at Sotheby 's , and this video via USA Today shows more reaction from the auction . The ID buzz : The video posted by Banksy appears to be taken from the vantage point of a man who was pictured at Sotheby 's filming the scene . You can see an image of him at Lad Bible . He 's a middle - aged man with curly hair , and countless online speculators point out that he resembles a street artist named Robin Gunningham , one of the leading suspects in the who - is - Banksy question . " All of this is of course speculation but when it comes to Banksy , let 's face it , everything is , " observes a post about the auction 's " mystery man " at Sky News . ID buzz , II : Caroline Lang , chief of Sotheby 's Switzerland , posted an image of another man who appeared to be activating a remote - control device , and she identified him as Banksy , reports the New York Times . Alas , the account is private and the photo unavailable . Inside job ? Sotheby 's swears it was n't in on the stunt , but some are skeptical . One dealer tells the Times that he pointed out to staff before the auction that the print 's frame was weirdly large , but they had no explanation . " If the upper management knew , I ca n't speculate . " Plus , the print was <placed(34)> in a relatively hard - to - access viewing spot before the <auction(39)> , then was <sold(5)> dead last in the 67 - item sale , which was " odd , " he says . Good question : If Banksy did indeed embed the shredder a " few years ago , " would n't the battery have needed replacement since then ? So wonders Scott Reyburn at the Times . The irony : Banksy may have been making a statement about what he views as absurd prices for his work , but the stunt likely increased the value of the print , writes Leonid Bershidsky at Bloomberg . Of course , Banksy surely knew that would happen , this being only his " latest contribution to the empirical study of the value of art . " That 's where his true genius lies , writes Bershidsky . Another take : Sebastian Smee at the Washington Post also digs into Banksy 's motivations and theme of destruction in avant - garde art . So what 's the main problem in all this ? " Is it a system that values art in monetary terms in order for it to be exchanged on the market ? Or is it a system in thrall to the currency of publicity and self - promotion ? If it ’s the latter , Banksy is deeply implicated . "

    Resolve all the relations between those pairs of events:
    stunt(12)/shredded(55), stunt(12)/sold(37), stunt(12)/revealed(23), stunt(12)/shredding(7), stunt(12)/built(30), stunt(12)/placed(34), stunt(12)/auction(39), stunt(12)/sold(5), shredded(55)/sold(37), shredded(55)/revealed(23), shredded(55)/shredding(7), shredded(55)/built(30), shredded(55)/placed(34), shredded(55)/auction(39), shredded(55)/sold(5), sold(37)/revealed(23), sold(37)/shredding(7), sold(37)/built(30), sold(37)/placed(34), sold(37)/auction(39), sold(37)/sold(5), revealed(23)/shredding(7), revealed(23)/built(30), revealed(23)/placed(34), revealed(23)/auction(39), revealed(23)/sold(5), shredding(7)/built(30), shredding(7)/placed(34), shredding(7)/auction(39), shredding(7)/sold(5), built(30)/placed(34), built(30)/auction(39), built(30)/sold(5), placed(34)/auction(39), placed(34)/sold(5), auction(39)/sold(5)

    Your expected output:
    stunt(12)/shredded(55)=equal, stunt(12)/sold(37)=after, stunt(12)/revealed(23)=before, stunt(12)/shredding(7)=equal, stunt(12)/built(30)=after, stunt(12)/placed(34)=after, stunt(12)/auction(39)=after, stunt(12)/sold(5)=after, shredded(55)/sold(37)=after, shredded(55)/revealed(23)=before, shredded(55)/shredding(7)=equal, shredded(55)/built(30)=after, shredded(55)/placed(34)=after, shredded(55)/auction(39)=after, shredded(55)/sold(5)=after, sold(37)/revealed(23)=before, sold(37)/shredding(7)=before, sold(37)/built(30)=after, sold(37)/placed(34)=after, sold(37)/auction(39)=after, sold(37)/sold(5)=equal, revealed(23)/shredding(7)=after, revealed(23)/built(30)=after, revealed(23)/placed(34)=after, revealed(23)/auction(39)=after, revealed(23)/sold(5)=after, shredding(7)/built(30)=after, shredding(7)/placed(34)=after, shredding(7)/auction(39)=after, shredding(7)/sold(5)=after, built(30)/placed(34)=before, built(30)/auction(39)=before, built(30)/sold(5)=before, placed(34)/auction(39)=before, placed(34)/sold(5)=before, auction(39)/sold(5)=before

    Output only the pair list with your predicted relations as indicated and without any additional information.

    Following is the input text with events for you to process:
    """
    return desc


def task_description_v2():
    desc = """
    Task Overview:
    You will analyze a text where events are marked with '<' and '>' symbols, and each event is identified with an ID shown in parentheses immediately after the event. A list of event pairs in the format event1(id1)/event2(id2) will be given. 
    Your task is to determine the temporal relationships between these pairs based on the starting times of the event mentions.

    Instructions for Determining Temporal Relationships:
    Before: If Event A starts before Event B. Example: "A traveler is <kidnapped(01)>, and the police officers <said(02)> The kidnapper is demanding money". 
    Since 'kidnapped' must have occurred before 'said', the relationship should be 'before'.
    After: If Event A starts after Event B. Using the previous example, the relationship for 'said(02)/kidnapped(01)' is 'after'.
    Equal: If Event A and B start simultaneously, for example: "They <filed(03)> objections, <claiming(04)> unfair treatment". Since both events refer to the same thing, they occured at the same time, thus, the relationship should be 'equal'.
    Uncertain: If the start times of Event A and B cannot be determined, for example: "I <ate(05)> and <drank(06)> at lunch". The sequence cannot be determined if I first started to eat or first started to drink as both are logically possible, the relationship should be 'uncertain'.

    Look for Temporal Indicators:
    Use explicit temporal indicators such as dates, times, and keywords like 'before', 'after', and 'at the same time' in the provided context to help determine relationships. 
    If no indicators are present, and logical deduction fails, mark the relationship as 'uncertain'.

    Output Format:
    Output the list of event pairs with the resolved temporal relationships in the format: event1(id1)/event2(id2)=relationship, ...

    A full example:
    Input:
    – " We 've just been Banksy'ed . " So declared Sotheby 's of London on Friday after a bizarre <stunt(12)> apparently pulled off by the elusive artist himself — one of his prints <shredded(55)> itself just after being <sold(37)> at auction for $ 1.4 million . Banksy has since <revealed(23)> how the <shredding(7)> of his 2006 " Girl With Balloon " took place , but some think the artist may have inadvertently revealed more than that : perhaps an image of himself , or at the very least a close assistant . Details and developments : The ' how ' : Banksy posted a video online showing that he <built(30)> a remote - controlled shredder into the print 's frame " a few years ago " so he could destroy the work if it ever went up for auction . The video also captures the initial moments of that happening at Sotheby 's , and this video via USA Today shows more reaction from the auction . The ID buzz : The video posted by Banksy appears to be taken from the vantage point of a man who was pictured at Sotheby 's filming the scene . You can see an image of him at Lad Bible . He 's a middle - aged man with curly hair , and countless online speculators point out that he resembles a street artist named Robin Gunningham , one of the leading suspects in the who - is - Banksy question . " All of this is of course speculation but when it comes to Banksy , let 's face it , everything is , " observes a post about the auction 's " mystery man " at Sky News . ID buzz , II : Caroline Lang , chief of Sotheby 's Switzerland , posted an image of another man who appeared to be activating a remote - control device , and she identified him as Banksy , reports the New York Times . Alas , the account is private and the photo unavailable . Inside job ? Sotheby 's swears it was n't in on the stunt , but some are skeptical . One dealer tells the Times that he pointed out to staff before the auction that the print 's frame was weirdly large , but they had no explanation . " If the upper management knew , I ca n't speculate . " Plus , the print was <placed(34)> in a relatively hard - to - access viewing spot before the <auction(39)> , then was <sold(5)> dead last in the 67 - item sale , which was " odd , " he says . Good question : If Banksy did indeed embed the shredder a " few years ago , " would n't the battery have needed replacement since then ? So wonders Scott Reyburn at the Times . The irony : Banksy may have been making a statement about what he views as absurd prices for his work , but the stunt likely increased the value of the print , writes Leonid Bershidsky at Bloomberg . Of course , Banksy surely knew that would happen , this being only his " latest contribution to the empirical study of the value of art . " That 's where his true genius lies , writes Bershidsky . Another take : Sebastian Smee at the Washington Post also digs into Banksy 's motivations and theme of destruction in avant - garde art . So what 's the main problem in all this ? " Is it a system that values art in monetary terms in order for it to be exchanged on the market ? Or is it a system in thrall to the currency of publicity and self - promotion ? If it ’s the latter , Banksy is deeply implicated . "

    Pairs:
    stunt(12)/shredded(55), stunt(12)/sold(37), stunt(12)/revealed(23), stunt(12)/shredding(7), stunt(12)/built(30), stunt(12)/placed(34), stunt(12)/auction(39), stunt(12)/sold(5), shredded(55)/sold(37), shredded(55)/revealed(23), shredded(55)/shredding(7), shredded(55)/built(30), shredded(55)/placed(34), shredded(55)/auction(39), shredded(55)/sold(5), sold(37)/revealed(23), sold(37)/shredding(7), sold(37)/built(30), sold(37)/placed(34), sold(37)/auction(39), sold(37)/sold(5), revealed(23)/shredding(7), revealed(23)/built(30), revealed(23)/placed(34), revealed(23)/auction(39), revealed(23)/sold(5), shredding(7)/built(30), shredding(7)/placed(34), shredding(7)/auction(39), shredding(7)/sold(5), built(30)/placed(34), built(30)/auction(39), built(30)/sold(5), placed(34)/auction(39), placed(34)/sold(5), auction(39)/sold(5)

    The expected correct output should be:
    stunt(12)/shredded(55)=equal, stunt(12)/sold(37)=after, stunt(12)/revealed(23)=before, stunt(12)/shredding(7)=equal, stunt(12)/built(30)=after, stunt(12)/placed(34)=after, stunt(12)/auction(39)=after, stunt(12)/sold(5)=after, shredded(55)/sold(37)=after, shredded(55)/revealed(23)=before, shredded(55)/shredding(7)=equal, shredded(55)/built(30)=after, shredded(55)/placed(34)=after, shredded(55)/auction(39)=after, shredded(55)/sold(5)=after, sold(37)/revealed(23)=before, sold(37)/shredding(7)=before, sold(37)/built(30)=after, sold(37)/placed(34)=after, sold(37)/auction(39)=after, sold(37)/sold(5)=equal, revealed(23)/shredding(7)=after, revealed(23)/built(30)=after, revealed(23)/placed(34)=after, revealed(23)/auction(39)=after, revealed(23)/sold(5)=after, shredding(7)/built(30)=after, shredding(7)/placed(34)=after, shredding(7)/auction(39)=after, shredding(7)/sold(5)=after, built(30)/placed(34)=before, built(30)/auction(39)=before, built(30)/sold(5)=before, placed(34)/auction(39)=before, placed(34)/sold(5)=before, auction(39)/sold(5)=before

    Explanation on the output pairs:
    stunt(12) and shredded(55) are the same thing, there for happened at the same time.
    the stunt(12) happened after the painting was sold(37).
    the stunt(12) happened before Banksy revealed(23) how the shredding(7) took place.
    the stunt(12) happened after Banksy built(30) the remote control to activate the shredding of the painting.
    ....

    Your output should only be the pair list with your predicted relations without any additional information.

    Following is the input text with events for you to process and predict:
    """
    return desc


def task_description_v3():
    desc = """
    Task Overview:
    You will analyze a text where events are marked with '<' and '>' symbols, and each event is identified with an ID shown in parentheses immediately after the event. A list of event pairs in the format event1(id1)/event2(id2) will be given. 
    Your task is to determine the temporal relationships between these pairs based on the starting times of the event mentions.
    The temporal relation can be one of {before, after, equal, uncertain (for when the temporal relation cannot be determine from the context)}.

    Output Format:
    Output the list of event pairs with the resolved temporal relationships in the format: event1(id1)/event2(id2)=relationship, ...

    Example:
    Input:
    – " We 've just been Banksy'ed . " So declared Sotheby 's of London on Friday after a bizarre <stunt(12)> apparently pulled off by the elusive artist himself — one of his prints <shredded(55)> itself just after being <sold(37)> at auction for $ 1.4 million . Banksy has since <revealed(23)> how the <shredding(7)> of his 2006 " Girl With Balloon " took place , but some think the artist may have inadvertently revealed more than that : perhaps an image of himself , or at the very least a close assistant . Details and developments : The ' how ' : Banksy posted a video online showing that he <built(30)> a remote - controlled shredder into the print 's frame " a few years ago " so he could destroy the work if it ever went up for auction . The video also captures the initial moments of that happening at Sotheby 's , and this video via USA Today shows more reaction from the auction . The ID buzz : The video posted by Banksy appears to be taken from the vantage point of a man who was pictured at Sotheby 's filming the scene . You can see an image of him at Lad Bible . He 's a middle - aged man with curly hair , and countless online speculators point out that he resembles a street artist named Robin Gunningham , one of the leading suspects in the who - is - Banksy question . " All of this is of course speculation but when it comes to Banksy , let 's face it , everything is , " observes a post about the auction 's " mystery man " at Sky News . ID buzz , II : Caroline Lang , chief of Sotheby 's Switzerland , posted an image of another man who appeared to be activating a remote - control device , and she identified him as Banksy , reports the New York Times . Alas , the account is private and the photo unavailable . Inside job ? Sotheby 's swears it was n't in on the stunt , but some are skeptical . One dealer tells the Times that he pointed out to staff before the auction that the print 's frame was weirdly large , but they had no explanation . " If the upper management knew , I ca n't speculate . " Plus , the print was <placed(34)> in a relatively hard - to - access viewing spot before the <auction(39)> , then was <sold(5)> dead last in the 67 - item sale , which was " odd , " he says . Good question : If Banksy did indeed embed the shredder a " few years ago , " would n't the battery have needed replacement since then ? So wonders Scott Reyburn at the Times . The irony : Banksy may have been making a statement about what he views as absurd prices for his work , but the stunt likely increased the value of the print , writes Leonid Bershidsky at Bloomberg . Of course , Banksy surely knew that would happen , this being only his " latest contribution to the empirical study of the value of art . " That 's where his true genius lies , writes Bershidsky . Another take : Sebastian Smee at the Washington Post also digs into Banksy 's motivations and theme of destruction in avant - garde art . So what 's the main problem in all this ? " Is it a system that values art in monetary terms in order for it to be exchanged on the market ? Or is it a system in thrall to the currency of publicity and self - promotion ? If it ’s the latter , Banksy is deeply implicated . "

    Pairs:
    stunt(12)/shredded(55), stunt(12)/sold(37), stunt(12)/revealed(23), stunt(12)/shredding(7), stunt(12)/built(30), stunt(12)/placed(34), stunt(12)/auction(39), stunt(12)/sold(5), shredded(55)/sold(37), shredded(55)/revealed(23), shredded(55)/shredding(7), shredded(55)/built(30), shredded(55)/placed(34), shredded(55)/auction(39), shredded(55)/sold(5), sold(37)/revealed(23), sold(37)/shredding(7), sold(37)/built(30), sold(37)/placed(34), sold(37)/auction(39), sold(37)/sold(5), revealed(23)/shredding(7), revealed(23)/built(30), revealed(23)/placed(34), revealed(23)/auction(39), revealed(23)/sold(5), shredding(7)/built(30), shredding(7)/placed(34), shredding(7)/auction(39), shredding(7)/sold(5), built(30)/placed(34), built(30)/auction(39), built(30)/sold(5), placed(34)/auction(39), placed(34)/sold(5), auction(39)/sold(5)

    The expected correct output should be:
    stunt(12)/shredded(55)=equal, stunt(12)/sold(37)=after, stunt(12)/revealed(23)=before, stunt(12)/shredding(7)=equal, stunt(12)/built(30)=after, stunt(12)/placed(34)=after, stunt(12)/auction(39)=after, stunt(12)/sold(5)=after, shredded(55)/sold(37)=after, shredded(55)/revealed(23)=before, shredded(55)/shredding(7)=equal, shredded(55)/built(30)=after, shredded(55)/placed(34)=after, shredded(55)/auction(39)=after, shredded(55)/sold(5)=after, sold(37)/revealed(23)=before, sold(37)/shredding(7)=before, sold(37)/built(30)=after, sold(37)/placed(34)=after, sold(37)/auction(39)=after, sold(37)/sold(5)=equal, revealed(23)/shredding(7)=after, revealed(23)/built(30)=after, revealed(23)/placed(34)=after, revealed(23)/auction(39)=after, revealed(23)/sold(5)=after, shredding(7)/built(30)=after, shredding(7)/placed(34)=after, shredding(7)/auction(39)=after, shredding(7)/sold(5)=after, built(30)/placed(34)=before, built(30)/auction(39)=before, built(30)/sold(5)=before, placed(34)/auction(39)=before, placed(34)/sold(5)=before, auction(39)/sold(5)=before

    Explanation on the reasoning behind some of the pairs in the example above:
    stunt(12) and shredded(55) are the same thing, there for happened at the same time.
    the stunt(12) happened after the painting was sold(37).
    the stunt(12) happened before Banksy revealed(23) how the shredding(7) took place.
    the stunt(12) happened after Banksy built(30) the remote control to activate the shredding of the painting.
    the painting was sold(37) after the auction(39) started (sold happened during the auction).

    Your output should be all the pairs in the list with your predicted relations without any additional information.

    Following is the input text with events for you to process and predict:
    """
    return desc


def task_description_v4():
    desc = """
    Task Overview:
    You will analyze a text where events are marked with '<' and '>' symbols, and each event is identified with an ID shown in parentheses immediately after the event. A list of event pairs in the format event1(id1)/event2(id2) will be given. 
    Your task is to determine the temporal relationships between these pairs based on the starting times of the event mentions.
    The temporal relation can be one of {before, after, equal, uncertain (for when the temporal relation cannot be determine from the context)}.

    Output Format:
    Output the list of event pairs with the resolved temporal relationships in the format: event1(id1)/event2(id2)=relationship, ...

    Example:
    Input:
    – " We 've just been Banksy'ed . " So declared Sotheby 's of London on Friday after a bizarre <stunt(12)> apparently pulled off by the elusive artist himself — one of his prints <shredded(55)> itself just after being <sold(37)> at auction for $ 1.4 million . Banksy has since <revealed(23)> how the <shredding(7)> of his 2006 " Girl With Balloon " took place , but some think the artist may have inadvertently revealed more than that : perhaps an image of himself , or at the very least a close assistant . Details and developments : The ' how ' : Banksy posted a video online showing that he <built(30)> a remote - controlled shredder into the print 's frame " a few years ago " so he could destroy the work if it ever went up for auction . The video also captures the initial moments of that happening at Sotheby 's , and this video via USA Today shows more reaction from the auction . The ID buzz : The video posted by Banksy appears to be taken from the vantage point of a man who was pictured at Sotheby 's filming the scene . You can see an image of him at Lad Bible . He 's a middle - aged man with curly hair , and countless online speculators point out that he resembles a street artist named Robin Gunningham , one of the leading suspects in the who - is - Banksy question . " All of this is of course speculation but when it comes to Banksy , let 's face it , everything is , " observes a post about the auction 's " mystery man " at Sky News . ID buzz , II : Caroline Lang , chief of Sotheby 's Switzerland , posted an image of another man who appeared to be activating a remote - control device , and she identified him as Banksy , reports the New York Times . Alas , the account is private and the photo unavailable . Inside job ? Sotheby 's swears it was n't in on the stunt , but some are skeptical . One dealer tells the Times that he pointed out to staff before the auction that the print 's frame was weirdly large , but they had no explanation . " If the upper management knew , I ca n't speculate . " Plus , the print was <placed(34)> in a relatively hard - to - access viewing spot before the <auction(39)> , then was <sold(5)> dead last in the 67 - item sale , which was " odd , " he says . Good question : If Banksy did indeed embed the shredder a " few years ago , " would n't the battery have needed replacement since then ? So wonders Scott Reyburn at the Times . The irony : Banksy may have been making a statement about what he views as absurd prices for his work , but the stunt likely increased the value of the print , writes Leonid Bershidsky at Bloomberg . Of course , Banksy surely knew that would happen , this being only his " latest contribution to the empirical study of the value of art . " That 's where his true genius lies , writes Bershidsky . Another take : Sebastian Smee at the Washington Post also digs into Banksy 's motivations and theme of destruction in avant - garde art . So what 's the main problem in all this ? " Is it a system that values art in monetary terms in order for it to be exchanged on the market ? Or is it a system in thrall to the currency of publicity and self - promotion ? If it ’s the latter , Banksy is deeply implicated . "

    Pairs:
    stunt(12)/shredded(55), stunt(12)/sold(37), stunt(12)/revealed(23), stunt(12)/shredding(7), stunt(12)/built(30), stunt(12)/placed(34), stunt(12)/auction(39), stunt(12)/sold(5), shredded(55)/sold(37), shredded(55)/revealed(23), shredded(55)/shredding(7), shredded(55)/built(30), shredded(55)/placed(34), shredded(55)/auction(39), shredded(55)/sold(5), sold(37)/revealed(23), sold(37)/shredding(7), sold(37)/built(30), sold(37)/placed(34), sold(37)/auction(39), sold(37)/sold(5), revealed(23)/shredding(7), revealed(23)/built(30), revealed(23)/placed(34), revealed(23)/auction(39), revealed(23)/sold(5), shredding(7)/built(30), shredding(7)/placed(34), shredding(7)/auction(39), shredding(7)/sold(5), built(30)/placed(34), built(30)/auction(39), built(30)/sold(5), placed(34)/auction(39), placed(34)/sold(5), auction(39)/sold(5)

    The expected correct output should be:
    stunt(12)/shredded(55)=equal, stunt(12)/sold(37)=after, stunt(12)/revealed(23)=before, stunt(12)/shredding(7)=equal, stunt(12)/built(30)=after, stunt(12)/placed(34)=after, stunt(12)/auction(39)=after, stunt(12)/sold(5)=after, shredded(55)/sold(37)=after, shredded(55)/revealed(23)=before, shredded(55)/shredding(7)=equal, shredded(55)/built(30)=after, shredded(55)/placed(34)=after, shredded(55)/auction(39)=after, shredded(55)/sold(5)=after, sold(37)/revealed(23)=before, sold(37)/shredding(7)=before, sold(37)/built(30)=after, sold(37)/placed(34)=after, sold(37)/auction(39)=after, sold(37)/sold(5)=equal, revealed(23)/shredding(7)=after, revealed(23)/built(30)=after, revealed(23)/placed(34)=after, revealed(23)/auction(39)=after, revealed(23)/sold(5)=after, shredding(7)/built(30)=after, shredding(7)/placed(34)=after, shredding(7)/auction(39)=after, shredding(7)/sold(5)=after, built(30)/placed(34)=before, built(30)/auction(39)=before, built(30)/sold(5)=before, placed(34)/auction(39)=before, placed(34)/sold(5)=before, auction(39)/sold(5)=before

    Your output should be all the pairs in the list with your predicted relations without any additional information.

    Following is the input text with events for you to process and predict:
    """
    return desc
