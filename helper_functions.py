
# Distinguish categorical and continuous features
def determine_type_of_feature(df):
    feature_types = []
    n_unique_values_treshold = 15
    for feature in df.columns:
        if feature != "label":
            unique_values = df[feature].unique()
            example_value = unique_values[0]

            if (isinstance(example_value, str)) or (len(unique_values) <= n_unique_values_treshold):
                feature_types.append("categorical")
            else:
                feature_types.append("continuous")

    return feature_types

# Determine local time
def localtime_in_sec(localtime):
    time = localtime()
    hour = time.tm_hour * 3600
    minute = time.tm_min * 60
    second = time.tm_sec
    localtime_sec = hour + minute + second

    return localtime_sec
