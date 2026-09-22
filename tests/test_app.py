"""Exercise the actual Streamlit screens and controls without a browser driver."""
import pytest
from streamlit.testing.v1 import AppTest
from preprocessing import ROOT

pytestmark = pytest.mark.skipif(
    not (ROOT / "models" / "gradient_boosting.joblib").exists(),
    reason="Run train.py first (the source dataset is bundled)")


def button(app, label):
    return next(item for item in app.button if item.label == label)


def test_all_pages_render_and_replay_unsubscribes():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    for page in ["Model lab", "About the paper", "Event replay"]:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, page
    button(app, "Subscribe").click().run()
    button(app, "Advance 10").click().run()
    assert not app.exception
    assert app.session_state["replay"].analytics.received == 10
    assert len(app.session_state["replay"].analytics.resolved) > 0
    button(app, "Unsubscribe").click().run()
    button(app, "Next event").click().run()
    assert app.session_state["replay"].analytics.received == 10
    assert app.session_state["replay"].log[-1]["delivered"] is False
    button(app, "Reset replay").click().run()
    assert app.session_state["replay"].cursor == 0
    assert not app.exception
