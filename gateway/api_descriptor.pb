
�
completion.protollmgoogle/api/annotations.protogoogle/api/field_behavior.proto.protoc-gen-openapiv2/options/annotations.proto"|
Usage#

completion_tokens (RcompletionTokens!
total_tokens (RtotalTokens"�
CompletionRequest
model (	B�ARmodel
prompt (	B�ARprompt

max_tokens (
temperature (Rtemperature
top_p (RtopP
n (
stream (Rstream
logprobs	 (
echo
 (Recho)
presence_penalty (RpresencePenalty+
frequency_penalty
best_of (
user (	Ruser"s
Choice
text (	Rtext
logprobs (Rlogprobs
index (

CompletionResponse
id (	Rid
object (	Robject
created (Rcreated
model (	Rmodel%
choices (2.llm.ChoiceRchoices 
usage (2
.llm.UsageRusage";
ChatMessage
role (	Rrole
content (	Rcontent"�
ChatRequest
model (	B�ARmodel2
messages (2.llm.ChatMessageB�ARmessages 
temperature (Rtemperature
top_p (RtopP
n (
stream (Rstream

max_tokens
 (R	maxTokens)
presence_penalty (RpresencePenalty+
frequency_penalty (RfrequencyPenalty
user (	Ruser"�

ChatChoice
index (Rindex&
delta (2.llm.ChatMessageRdelta*
message (2.llm.ChatMessageRmessage#

ChatResponse
id (	Rid
created (Rcreated
model (	Rmodel)
choices (2.llm.ChatChoiceRchoices 
usage (2
.llm.UsageRusage2�

Completiono
Complete.llm.CompletionRequest.llm.CompletionResponse"0�A:text/event-stream���"/v1/completions:*0d
Chat.llm.ChatRequest.llm.ChatResponse"5�A:text/event-stream���"/v1/chat/completions:*0B*Z(github.com/vectorch-ai/scalellm;scalellmbproto3