import googleapiclient.discovery
import pandas as pd
import re

api_key = 'AIzaSyALscOczU9-MfbsGnahhD82zXjgQNaXG4s'


def get_comments(video_id):
    youtube = googleapiclient.discovery.build('youtube', 'v3', developerKey=api_key)

    comments = []
    request = youtube.commentThreads().list(part='snippet', videoId=video_id, textFormat='plainText', maxResults=100)

    while request:
        response = request.execute()
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            author = item['snippet']['topLevelComment']['snippet']['authorDisplayName']
            like_count = item['snippet']['topLevelComment']['snippet']['likeCount']
            comments.append({'author': author, 'comment': comment, 'like_count': like_count})

        request = youtube.commentThreads().list_next(request, response)

    return pd.DataFrame(comments)


# Örnek video ID'si
video_id = '5ON3WiM0k34'
comments_df = get_comments(video_id)
comments_df.to_csv('comments.csv', index=False)
