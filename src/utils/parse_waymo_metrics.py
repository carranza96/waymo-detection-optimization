from absl import app
from absl import flags
import pandas as pd

FLAGS = flags.FLAGS
flags.DEFINE_string('metrics_file', None, 'Text file containing metrics in raw format')

## TODO: calculate Mean value
def main(_):
    f = open(FLAGS.metrics_file, 'rb')

    num_examples = f.readline().decode().split(" ")[0]

    rows = [{'Metric Breakdown': 'NUMBER OF EXAMPLES', 'mAP': num_examples}]
    for line in f.readlines()[1:]:
        sp = line.decode().split(":")
        breakdown = sp[0]
        map_metric = sp[1].split("]")[0].split(" [mAP ")[1]
        if 'OBJECT_TYPE_TYPE_SIGN' not in breakdown:
            rows.append({'Metric Breakdown': breakdown, 'mAP': map_metric})

    f.close()

    f = open(FLAGS.metrics_file, 'w')
    metrics_df = pd.DataFrame(rows, columns=['Metric Breakdown', 'mAP'])
    metrics_df.to_csv(f, index=False)
    f.close()


if __name__ == '__main__':
    app.run(main)

