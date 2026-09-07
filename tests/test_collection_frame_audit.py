import sys, tempfile, json, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/teleop_free/tools'))
from collection_frame_audit import install

class FrameReceiptTest(unittest.TestCase):
    def test_receipts_follow_successful_rows_and_reset_episode_index(self):
        class Collector:
            current_episode_index=0
            latest_proprio_msg={'index':7,'ros_timestamp':10.}
            latest_image_msg={'timestamps':{'ego_view':9.98}}
            def _add_data_frame_sonic(self,t_start):
                self._latest_priv={'seq':12}
                return 'unchanged'
        with tempfile.TemporaryDirectory() as td:
            file=Path(td)/'audit.jsonl';stream=install(Collector,file)
            c=Collector()
            self.assertEqual(c._add_data_frame_sonic(0),'unchanged')
            c._add_data_frame_sonic(0);c.current_episode_index=1;c._add_data_frame_sonic(0)
            stream.close();rows=list(map(json.loads,file.read_text().splitlines()))
            self.assertEqual([r['row_index'] for r in rows],[0,1,0])
            self.assertEqual(rows[0]['priv_seq'],12)
            self.assertEqual(rows[0]['image_publish_wall'],{'ego_view':9.98})
            self.assertEqual(c.latest_proprio_msg,{'index':7,'ros_timestamp':10.})
