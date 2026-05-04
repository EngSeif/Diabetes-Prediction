class FeatureSelection:

    def __init__(self, df):
        self.df = df

    def drop_features(self):
        to_drop = [
            "age",
            "blood_glucose_level",
            "age_bmi_interaction",
            "age_hba1c_interaction",
            "hypertension",
            #"bmi_category",
            #"hba1c_category",
            #"blood_glucose_category",
        ]
        cleaned = self.df.drop(columns=to_drop)
        return cleaned
