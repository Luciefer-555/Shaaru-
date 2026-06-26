Created At: 2026-06-13T11:14:25Z
Completed At: 2026-06-13T11:14:25Z
File Path: `file:///C:/Users/saipr/Downloads/Shaaru/shaaru_brain.py`
Total Lines: 948
Total Bytes: 40332
Showing lines 750 to 885
The following code has been modified to include a line number before every line, in the format: <line_number>: <original_line>. Please note that any changes targeting the original code should remove the line number, colon, and leading space.
750:         # Tiered model routing
751:         query_tier = classify_query_complexity(user_message)
752: 
753:         if image_b64:
754:             active_model = "meta/llama-3.2-90b-vision-instruct"
755:             headers = {
756:                 "Authorization": f"Bearer {os.getenv('NVIDIA_API_KEY')}",
757:                 "Content-Type": "application/json",
758:             }
759:             vision_messages = [{"role": "system", "content": system_content}]
760:             for msg in chat_history[-6:]:
761:                 vision_messages.append(msg)
762:             vision_messages.append({
763:                 "role": "user",
764:                 "content": [
765:                     {
766:                         "type": "image_url",
767:                         "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
768:                     },
769:                     {"type": "text", "text": user_message}
770:                 ]
771:             })
772:             
773:             shaaru_response = ""
774:             
775:             # Try 90b vision first
776:             try:
777:                 payload = {
778:                     "model": active_model,
779:                     "messages": vision_messages,
780:                     "temperature": 0.4,
781:                     "max_tokens": 1200,
782:                 }
783:                 resp = requests.post(
784:                     "https://integrate.api.nvidia.com/v1/chat/completions",
785:                     headers=headers,
786:                     json=payload,
787:                     timeout=90,
7
<truncated 2776 bytes>
txt}")
841:                     shaaru_response = "send that again, something dropped on my end"
842: 
843:         elif query_tier == "complex":
844:             # Complex styling — DeepSeek with medium reasoning
845:             completion = nvidia_call(
846:                 nvidia_client,
847:                 model="deepseek-ai/deepseek-r1",
848:                 messages=messages,
849:                 temperature=0.4,
850:                 max_tokens=1200,
851:                 extra_body={
852:                     "chat_template_kwargs": {
853:                         "thinking": True,
854:                         "reasoning_effort": "medium"
855:                     }
856:                 }
857:             )
858:             shaaru_response = completion.choices[0].message.content.strip()
859: 
860:         else:
861:             # Simple — fast 70B
862:             completion = nvidia_call(
863:                 nvidia_client,
864:                 model="meta/llama-3.1-70b-instruct",
865:                 messages=messages,
866:                 temperature=0.4,
867:                 max_tokens=1200,
868:             )
869:             shaaru_response = completion.choices[0].message.content.strip()
870: 
871:         # Log nudge
872:         try:
873:             if "NUDGE OPPORTUNITY:" in session_context:
874:                 log_nudge(user_id, "chat_trigger", shaaru_response)
875:         except Exception:
876:             pass
877: 
878:         return shaaru_response
879: 
880:     except Exception as e:
881:         print(f"[FAIL] chat_with_riley: {e}")
882:         return "send that again, something dropped on my end"
883: 
884: 
885: # ── chat_with_riley_cached ────────────────────────────────────────────────────
The above content does NOT show the entire file contents. If you need to view any lines of the file which were not shown to complete your task, call this tool again to view those lines.
