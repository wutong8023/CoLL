#!/usr/bin/env python
#
# Copyright the CoLL team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
# Intro: 
# Author: Tongtong Wu
# Time: Aug 4, 2021
"""

from argparse import ArgumentParser
from torch.utils.data import DataLoader, Dataset

from coll.dataset.base_dataset import add_base_dataset_args, BaseDataset


# modularized arguments management
def add_clinc150_dataset_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the models.
    :param parser: the parser instance
    """
    add_base_dataset_args(parser)


class Clinc150(BaseDataset):
    def __init__(self, args):
        super(Clinc150, self).__init__(args)
        # file_path
        self.file_path = ""
        
        # link: google file id
        self.file_code = "13-MkxdzWcW_FkYNyBcU_FUh1WFplA3Xq"
        
        # dataset```
        self.data = []
        self.labelled_data = []
        self.unlabelled_data = []
        
        # meta information
        self.label_dic = {
            "banking": [
                "transfer",
                "transactions",
                "balance",
                "freeze_account",
                "pay_bill",
                "bill_balance",
                "bill_due",
                "interest_rate",
                "routing",
                "min_payment",
                "order_checks",
                "pin_change",
                "report_fraud",
                "account_blocked",
                "spending_history"
            ],
            "credit_cards": [
                "credit_score",
                "report_lost_card",
                "credit_limit",
                "rewards_balance",
                "new_card",
                "application_status",
                "card_declined",
                "international_fees",
                "apr",
                "redeem_rewards",
                "credit_limit_change",
                "damaged_card",
                "replacement_card_duration",
                "improve_credit_score",
                "expiration_date"
            ],
            "kitchen_&_dining": [
                "recipe",
                "restaurant_reviews",
                "calories",
                "nutrition_info",
                "restaurant_suggestion",
                "ingredients_list",
                "ingredient_substitution",
                "cook_time",
                "food_last",
                "meal_suggestion",
                "restaurant_reservation",
                "confirm_reservation",
                "how_busy",
                "cancel_reservation",
                "accept_reservations"
            ],
            "home": [
                "shopping_list",
                "shopping_list_update",
                "next_song",
                "play_music",
                "update_playlist",
                "todo_list",
                "todo_list_update",
                "calendar",
                "calendar_update",
                "what_song",
                "order",
                "order_status",
                "reminder",
                "reminder_update",
                "smart_home"
            ],
            "auto_&_commute": [
                "traffic",
                "directions",
                "gas",
                "gas_type",
                "distance",
                "current_location",
                "mpg",
                "oil_change_when",
                "oil_change_how",
                "jump_start",
                "uber",
                "schedule_maintenance",
                "last_maintenance",
                "tire_pressure",
                "tire_change"
            ],
            "travel": [
                "book_flight",
                "book_hotel",
                "car_rental",
                "travel_suggestion",
                "travel_alert",
                "travel_notification",
                "carry_on",
                "timezone",
                "vaccines",
                "translate",
                "flight_status",
                "international_visa",
                "lost_luggage",
                "plug_type",
                "exchange_rate"
            ],
            "utility": [
                "time",
                "alarm",
                "share_location",
                "find_phone",
                "weather",
                "text",
                "spelling",
                "make_call",
                "timer",
                "date",
                "calculator",
                "measurement_conversion",
                "flip_coin",
                "roll_dice",
                "definition"
            ],
            "work": [
                "direct_deposit",
                "pto_request",
                "taxes",
                "payday",
                "w2",
                "pto_balance",
                "pto_request_status",
                "next_holiday",
                "insurance_change",
                "schedule_meeting",
                "pto_used",
                "meeting_schedule",
                "rollover_401k",
                "income",
                "insurance"
            ],
            "small_talk": [
                "greeting",
                "goodbye",
                "tell_joke",
                "where_are_you_from",
                "how_old_are_you",
                "what_is_your_name",
                "who_made_you",
                "thank_you",
                "what_can_i_ask_you",
                "what_are_your_hobbies",
                "do_you_have_pets",
                "are_you_a_bot",
                "meaning_of_life",
                "who_do_you_work_for",
                "fun_fact"
            ],
            "meta": [
                "change_ai_name",
                "change_user_name",
                "cancel",
                "user_name",
                "reset_settings",
                "whisper_mode",
                "repeat",
                "no",
                "yes",
                "maybe",
                "change_language",
                "change_accent",
                "change_volume",
                "change_speed",
                "sync_device"
            ]
        }

        self.target = None
        self.tokenizer = None
        self.label2id = {}
        
        # whether need development dataset
        self.require_dev = False
        
        # learning paradigm
        self.paradigm = "supervised"
        if self.paradigm == "supervised":
            self.data = self.labelled_data
        else:
            self.data = self.unlabelled_data
        pass
    

    def download_data(self):
        """
        download data from the google drive.
        """
        pass
    
    def load_data(self):
        """
        load data from files.
        """
        pass
    
    def filter_data(self, upper_bound: int, lower_bound: int):
        """
        filter data
        """
        pass
    
    def split_data(self, data):
        """
        split data into train / test / validation
        """
        pass
    
    def set_data(self, data, target=None, label2id=None):
        if label2id is not None:
            self.label2id = label2id
        self.data = data
        if target is None:
            self.targets = np.array([self.label2id[item["y_id"]] for item in self.data])
    
    def preprocess_data(self):
        """
        data preprocessing
        """
        pass
