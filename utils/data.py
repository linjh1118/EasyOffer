

def get_dummy_one_messages(with_image=False):
    print(f"getting dummy message. \nWith image: {with_image}")
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Who are you?"}],
        }
    ]
    if not with_image:
        return messages
    messages_with_image = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]
    return messages_with_image

