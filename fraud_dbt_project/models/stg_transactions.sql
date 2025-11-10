-- Selects and lightly cleans the columns needed for the ML model.
-- Complex feature engineering (age, distance) is done in the Python pipeline.

select
    -- We cast the datatypes here to ensure they are correct
    trans_date_trans_time::timestamp as transaction_timestamp,
    dob::date as date_of_birth,
    category::varchar as category,
    amt::float as amount,
    gender::varchar as gender,
    job::varchar as job,
    city_pop::integer as city_population,

    -- Location features
    lat::float as customer_latitude,
    long::float as customer_longitude, -- Note: original column name is 'long'
    merch_lat::float as merchant_latitude,
    merch_long::float as merchant_longitude,

    -- The target variable
    is_fraud::integer as is_fraud

from
    -- Reference the raw table defined in sources.yml
    {{ source('raw_data', 'raw_simulated_transactions') }}