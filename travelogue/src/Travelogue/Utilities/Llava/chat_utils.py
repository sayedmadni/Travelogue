#  -------------------------------------------------------------------------------------------------
#   Copyright (c) 2024.  SupportVectors AI Lab
#   This code is part of the training material, and therefore part of the intellectual property.
#   It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#   Use is limited to the duration and purpose of the training at SupportVectors.
#
#   Author: SupportVectors AI Training Team
#  -------------------------------------------------------------------------------------------------

import textwrap
class ChatHelper:
    def __init__(self, width=80):
        """
        Initialize the helper with a specified text width.
        
        Args:
            width (int): Maximum width for wrapping text. Default is 80.
        """
        self.width = width

    def chat_format(self, output_text):
        """
        Format the output text from the model into a readable chat format.
        
        Args:
            output_text (str): The raw output text from the model.
        
        Returns:
            str: Formatted chat output with user and assistant messages.
        """
        formatted_output = []
        # Split the output into individual messages based on "USER:" and "ASSISTANT:"
        messages = output_text.split("USER:")
        
        for message in messages:
            if not message.strip():  # Skip empty strings
                continue
            
            # Split into user and assistant parts
            if "ASSISTANT:" in message:
                user_part, assistant_part = message.split("ASSISTANT:")
                user_part = user_part.strip()
                assistant_part = assistant_part.strip()
                
                # Wrap the assistant's response
                wrapped_assistant = textwrap.fill(assistant_part, width=self.width)
                
                # Format the message
                formatted_output.append(f"USER:\n{user_part}\nASSISTANT:\n{wrapped_assistant}")
            else:
                # Handle cases where there's only a user message
                formatted_output.append(f"USER:\n{message.strip()}")
        
        # Join all formatted messages with double newlines for separation
        return "\n\n".join(formatted_output)

    def create_chat_template(self, user_query, previous_conversation=None, include_image=True):
        """
        Create a chat template from a user query, optionally including a previous conversation.
        
        Args:
            user_query (str): The user's text query.
            previous_conversation (list, optional): A list of previous conversation turns. Default is None.
            include_image (bool): Whether to include an image in the template. Default is True.
        
        Returns:
            list: A chat template in the format expected by multimodal models.
        """
        # Initialize the conversation with the previous conversation if provided
        conversation = previous_conversation.copy() if previous_conversation else []
        
        # Add the new user query to the conversation
        user_message = {
            "role": "user",
            "content": [
                {"type": "image"} if include_image else None,  # Include image placeholder if specified
                {"type": "text", "text": user_query},  # Include the user's text query
            ],
        }
        # Remove None values if include_image is False
        user_message["content"] = [item for item in user_message["content"] if item is not None]
        conversation.append(user_message)
        
        return conversation

    def convert_output_to_conversation(self, output_text, include_image=True):
        """
        Convert the raw output text from the model into a conversation chat template.
        
        Args:
            output_text (str): The raw output text from the model.
            include_image (bool): Whether to include an image in the template. Default is True.
        
        Returns:
            list: A conversation chat template in the format expected by multimodal models.
        """
        conversation = []
        # Split the output into individual messages based on "USER:" and "ASSISTANT:"
        messages = output_text.split("USER:")
        
        for message in messages:
            if not message.strip():  # Skip empty strings
                continue
            
            # Split into user and assistant parts
            if "ASSISTANT:" in message:
                user_part, assistant_part = message.split("ASSISTANT:")
                user_part = user_part.strip()
                assistant_part = assistant_part.strip()
                
                # Add the user message to the conversation
                user_message = {
                    "role": "user",
                    "content": [
                        {"type": "image"} if include_image else None,  # Include image placeholder if specified
                        {"type": "text", "text": user_part},  # Include the user's text query
                    ],
                }
                # Remove None values if include_image is False
                user_message["content"] = [item for item in user_message["content"] if item is not None]
                conversation.append(user_message)
                
                # Add the assistant message to the conversation
                assistant_message = {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": assistant_part}  # Include the assistant's response
                    ],
                }
                conversation.append(assistant_message)
            else:
                # Handle cases where there's only a user message
                user_message = {
                    "role": "user",
                    "content": [
                        {"type": "image"} if include_image else None,  # Include image placeholder if specified
                        {"type": "text", "text": message.strip()},  # Include the user's text query
                    ],
                }
                # Remove None values if include_image is False
                user_message["content"] = [item for item in user_message["content"] if item is not None]
                conversation.append(user_message)
        
        return conversation


