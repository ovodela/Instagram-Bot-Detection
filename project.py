import instaloader
from datetime import datetime
from itertools import dropwhile, takewhile
import csv
import os
import time

class GetInstagramProfile():
    def __init__(self) -> None:
        self.L = instaloader.Instaloader()

    def download_users_profile_picture(self,username):
        self.L.download_profile(username, profile_pic_only=True)

    def download_users_posts_with_periods(self,username):
        posts = instaloader.Profile.from_username(self.L.context, username).get_posts()
        SINCE = datetime(2021, 8, 28)
        UNTIL = datetime(2021, 9, 30)

        for post in takewhile(lambda p: p.date > SINCE, dropwhile(lambda p: p.date > UNTIL, posts)):
            self.L.download_post(post, username)

    def download_hastag_posts(self, hashtag):
        for post in instaloader.Hashtag.from_name(self.L.context, hashtag).get_posts():
            self.L.download_post(post, target='#'+hashtag)

    def get_users_followers(self,user_name):
        cls.L.login(input("input your username: "), input("input your password: ") ) 
        profile = instaloader.Profile.from_username(cls.L.context, "theairasian") #CHANGE FOR YOUR ACCOUNT
        #file = open("follower_names.txt","a+")
        followers = profile.get_followers()
        
        for follower in followers:
            followers = str(follower.followers)
            time.sleep(10)
            following = str(follower.followees)
            time.sleep(10)
            username = follower.username
            time.sleep(10)
            media_count = str(follower.mediacount)
            time.sleep(10)
            has_highlight = follower.has_highlight_reels
            time.sleep(10)
            has_url = follower.external_url
            time.sleep(10)
            biography_length = len(follower.biography)
            username_length = len(username)
            username_digits = 0
            for i in range(username_length):
                if(username[i].isdigit()):
                    username_digits += 1


            print(username + ", " + media_count + ", " + followers + ", " 
                + following + ", " + str(has_highlight) + ", " + str(has_url) + ", " 
                + str(biography_length) + ", " + str(username_length) + ", " + str(username_digits))
            time.sleep(10)

    def get_users_followings(self,user_name):
        '''Note: login required to get a profile's followings.'''
        self.L.login(input("input your username: "), input("input your password: ") ) 
        profile = instaloader.Profile.from_username(self.L.context, user_name)
        file = open("following_names.txt","a+")
        for followee in profile.get_followees():
            username = followee.username
            file.write(username + "\n")
            print(username)

    def get_post_comments(self,username):
        posts = instaloader.Profile.from_username(self.L.context, username).get_posts()
        for post in posts:
            for comment in post.get_comments():
                print("comment.id  : "+str(comment.id))
                print("comment.owner.username  : "+comment.owner.username)
                print("comment.text  : "+comment.text)
                print("comment.created_at_utc  : "+str(comment.created_at_utc))
                print("************************************************")

    def get_post_info_csv(self,username):
        with open(username+'.csv', 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            posts = instaloader.Profile.from_username(self.L.context, username).get_posts()
            for post in posts:
                print("post date: "+str(post.date))
                print("post profile: "+post.profile)
                print("post caption: "+str(post.caption))
                print("post location: "+str(post.location))
                
                posturl = "https://www.instagram.com/p/"+post.shortcode
                print("post url: "+posturl)
                writer.writerow(["post",post.mediaid, post.profile, post.caption, post.date, post.location, posturl,  post.typename, post.mediacount, post.caption_hashtags, post.caption_mentions, post.tagged_users, post.likes, post.comments,  post.title,  post.url ])
            
                for comment in post.get_comments():
                    writer.writerow(["comment",comment.id, comment.owner.username,comment.text,comment.created_at_utc])
                    print("comment username: "+comment.owner.username)
                    print("comment text: "+comment.text)
                    print("comment date : "+str(comment.created_at_utc))
                print("\n\n")


if __name__=="__main__":
    cls = GetInstagramProfile()
  
    cls.get_users_followers("theairasian") #change your name
    
    


        

