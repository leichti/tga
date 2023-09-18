from tga.data_loader import SavgolSmoother, NormalizeWithInitial


def smooth(segment):
    # smoothing
    segment.apply("m", SavgolSmoother(2001, 4), new_name="m_s")
    segment.apply("m_s", SavgolSmoother(1001, 4), new_name="m_s2")
    segment.apply("m_s2", NormalizeWithInitial())