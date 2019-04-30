import nidaqmx
import nidaqmx.stream_writers
import numpy as np

with nidaqmx.Task() as atask:
    with nidaqmx.Task() as dtask:
        atask.ao_channels.add_ao_voltage_chan("Dev1/ao0")
        dtask.do_channels.add_do_chan("Dev1/port0/line7")

        
        dtask.timing.samp_timing_type = nidaqmx.constants.SampleTimingType.SAMPLE_CLOCK
        atask.timing.samp_timing_type = nidaqmx.constants.SampleTimingType.SAMPLE_CLOCK
        dtask.timing.cfg_samp_clk_timing(1e5)
        atask.timing.cfg_samp_clk_timing(1e5)

        awriter = nidaqmx.stream_writers.AnalogSingleChannelWriter(atask.out_stream, auto_start=False)
        dwriter = nidaqmx.stream_writers.DigitalSingleChannelWriter(dtask.out_stream, auto_start=False)

        # 1 Megasample per second
        # atask.write(np.linspace(0.0, 5.0, 1000000), auto_start=True)
        high_ticks = 10
        low_ticks = 1000
        # atask.write(0, auto_start = False)
        # atask.write(0, auto_start = False)
        awriter.write_many_sample(np.concatenate([np.repeat(-10.0, high_ticks), np.repeat(10.0, high_ticks), np.repeat(-10.0, low_ticks//2)]))

        # dtask.write(0, auto_start = False)
        # dtask.write(0, auto_start = False)
        dwriter.write_many_sample_port_byte(np.concatenate([[np.uint8(0)], np.repeat(np.uint8(255), high_ticks), [np.uint8(0)]]))

        dtask.start()
        atask.start()

        dtask.wait_until_done()
        atask.wait_until_done()