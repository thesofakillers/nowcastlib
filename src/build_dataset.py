import os
import sys
import json
import pandas
import numpy

def get_cont_chunks(df, date_col, maxStep, minDuration):
    '''Find continuous blocks of time in a given array.
    '''
    timeArray = df[date_col]
    # indexes where a discontinuity in time occurs
    idxs = numpy.where(numpy.diff(timeArray) > maxStep)[0]
    if len(idxs) == 0:
        # trick: add a last time sample with a huge jump to make the routine
        # consider a contiguous block as a single block of data
        timeArray_hack = numpy.concatenate([timeArray, 1e45])
        numpy.where(numpy.diff(timeArray_hack) > maxStep)[0]
        return [0, timeArray.size]
    print("found {} discontinuities".format(len(idxs)))
   
    leftIdx = 0
    rightIdx = -1
    interval_list = list()
    for idx in idxs:
        rightIdx = idx
        duration = timeArray[rightIdx] - timeArray[leftIdx]
        if duration > minDuration:
            interv = pandas.Interval(timeArray[leftIdx], timeArray[rightIdx])
            interval_list.append(interv)
        leftIdx = rightIdx + 1
    intervals = pandas.arrays.IntervalArray(interval_list)
    return intervals

if __name__ == '__main__':
    
    # read in positional parameters: TODO, change by actual parse_args!
    config_path = sys.argv[1]
    
    # ugly configuration loading code
    config_file = open(config_path, 'r')
    ds_config = json.loads(config_file.read())
    data_config = ds_config['dataset']
    config_file.close()
    chunk_config = data_config.get("chunk_config")
    training_config = data_config.get("training_config")
    source_configs = data_config.get("data_sources")
    hdf5_path = os.path.join(chunk_config['path_to_hdf5'], chunk_config['tag'])
    sample_spacing_min = chunk_config['sample_spacing']
    min_chunk_duration_sec = chunk_config["min_chunk_duration_sec"]
    max_sync_block_dt_sec = chunk_config["max_delta_between_blocks_sec"]
    min_date = pandas.to_datetime(chunk_config.get('min_date', "1900-01-01T00:00:00"))
    max_date = pandas.to_datetime(chunk_config.get('max_date', "2100-12-31T00:00:00"))
    
    # load data
    dfs = list()
    cont_date_intervals = list()
    for source_name, source_info in source_configs.items():
        print("[INFO] loading source {}".format(source_name))
        fpath = source_info['file']
        field_list = source_info['field_list']
        cos_sin_fields = source_info.get('cos_sin_fields', None)
        date_col = source_info.get('date_time_column_name', 'Date time')
        date_fmt = source_info.get('date_time_column_format', "%Y-%m-%dT%H:%M:%S")
        data = pandas.read_csv(fpath)
        data = data.dropna()
        data = data.reset_index(drop=True)
        data['master_datetime'] = pandas.to_datetime(data[date_col], format=date_fmt)
        data = data.drop([date_col], axis=1)
        # transform fields to cos/sin
        if cos_sin_fields is not None:
            for func, pname in zip([numpy.cos, numpy.sin], ["Cosine", "Sine"]):
                for fname in cos_sin_fields:
                    new_fname = pname + " " + fname
                    field_data = data[fname]
                    if 'deg' in fname:
                        field_data = field_data*numpy.pi/180.0
                    data[new_fname] = field_data
        # find continuous chunks of data
        chunks = get_cont_chunks(
            data, 'master_datetime',
            pandas.Timedelta(max_sync_block_dt_sec, unit='s'), 
            pandas.Timedelta(min_chunk_duration_sec, unit='s'))
        dfs.append(data)
        cont_date_intervals.append(chunks)
    
    hdfs = pandas.HDFStore(hdf5_path)
    ik = 0 
    n_samples = 0
    data_overlaps = list()
    intervals_i = cont_date_intervals[0]
    for inter_i in intervals_i:
        n_source_overlaps = 0
        interval_overlaps = list()
        # check if interval overlaps with the rest of the data sources
        for intervals_j in cont_date_intervals[1::]:
            source_overlaps = list()
            overlap_mask = intervals_j.overlaps(inter_i)
            overlap_inter = intervals_j[overlap_mask]
            if len(overlap_inter) > 0:
                n_source_overlaps += 1 
                # find overlaps
                for overlap in overlap_inter:
                    o_left = max(inter_i.left, overlap.left)
                    o_right = min(inter_i.right, overlap.right)
                    source_overlaps.append(pandas.Interval(o_left, o_right, closed='neither'))
            interval_overlaps.append(source_overlaps)
        interval_overlaps.append(source_overlaps)
        # if true, interval overlaps with all data sources
        if n_source_overlaps != len(source_configs.keys()) - 1:
            continue
        # build dataframe with overlapping data from all sources
        resample_ok = True
        all_slices = list()
        for src_idx, src_os in enumerate(interval_overlaps):
            df = dfs[src_idx]
            # fix: use first slice only
            src_o = src_os[0]
            odf = df[(df['master_datetime'] > src_o.left) & (df['master_datetime'] < src_o.right)]
            odf = odf.set_index(pandas.DatetimeIndex(odf['master_datetime']))
            odf = odf.drop(['master_datetime'], axis=1)
            odf = odf.dropna()
            try:
                df_slice = odf.resample(sample_spacing_min, closed=None).bfill()
                all_slices.append(df_slice)
            except:
                resample_ok = False
                print("fail")
                break
        if resample_ok:
            synced_df = pandas.concat(all_slices, axis=1, join='inner')
            if len(synced_df) > 1:
                # transform datetime to cos/sin day
                datetime = synced_df.index.to_series()
                if datetime.iloc[0] < min_date:
                    continue
                if datetime.iloc[-1] > max_date:
                    continue
                print(datetime.iloc[0], "to", datetime.iloc[-1])
                sec_day = (datetime - datetime.dt.normalize())/pandas.Timedelta(seconds=1)
                cos_sec_day = numpy.cos(2*numpy.pi*sec_day.values/86400.0)
                sin_sec_day = numpy.sin(2*numpy.pi*sec_day.values/86400.0)
                synced_df['Cosine Day'] = cos_sec_day
                synced_df['Sine Day'] = sin_sec_day
                synced_df.to_hdf(hdfs, "chunk_{:d}".format(ik), format='table')
                n_samples += len(synced_df)
                print(n_samples)
        else:
            print("resampled failed, ignoring chunk")
        ik += 1
