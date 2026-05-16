"""
Fake Sri Lankan identity data generators.

All data generated here is entirely synthetic. Real personal identity documents
must NEVER be committed to this repository (see CLAUDE.md).

Usage:
    from scripts.dataset.fake_data import FakeSriLankan
    gen = FakeSriLankan(seed=42)
    print(gen.nic_number())
    print(gen.full_name())
    print(gen.address())
"""

from __future__ import annotations

import random
import string
from datetime import date, timedelta


# ── Sri Lankan name pools ─────────────────────────────────────────────────────

_SINHALA_GIVEN_NAMES = [
    "Lahiru", "Kasun", "Chamari", "Nimal", "Sunil", "Kavindi", "Dinesh",
    "Sachini", "Tharaka", "Amaya", "Chathura", "Dilhani", "Ishara",
    "Buddhika", "Malsha", "Nuwan", "Samantha", "Thilini", "Priyanka",
    "Randima", "Yashodha", "Madushanka", "Ruwan", "Dilshan", "Sanduni",
    "Hasitha", "Chaminda", "Nipuni", "Saman", "Rashmi", "Udara", "Menaka",
    "Dasun", "Roshani", "Janith", "Tharindi", "Harsha", "Shashini",
]

_TAMIL_GIVEN_NAMES = [
    "Kumar", "Priya", "Raj", "Anitha", "Selvam", "Kavitha", "Murali",
    "Geetha", "Suresh", "Vijay", "Ranjith", "Meena", "Kannan", "Nirmala",
    "Balan", "Saranya", "Siva", "Malathi", "Durai", "Thilaga",
]

_SINHALA_SURNAMES = [
    "Perera", "Silva", "Fernando", "Bandara", "Jayasinghe", "Rajapaksa",
    "Wickramasinghe", "Gunasekara", "Rathnayake", "Herath", "Gunawardena",
    "Dissanayake", "Weerasinghe", "Pathirana", "Liyanage", "Senanayake",
    "Amarasinghe", "Jayawardena", "Madhushanka", "Karunarathne", "Senevirathne",
    "Wijesuriya", "Abeywickrama", "Gamage", "Ranasinghe", "Kumarasinghe",
]

_TAMIL_SURNAMES = [
    "Balasingham", "Selvarajah", "Nadarajah", "Thambiayah", "Ratnasabapathy",
    "Mylvaganam", "Kathirgamanathan", "Nagendram", "Sivananthan", "Chandrakumar",
]

_DISTRICTS = [
    "Colombo", "Gampaha", "Kalutara", "Kandy", "Matale", "Nuwara Eliya",
    "Galle", "Matara", "Hambantota", "Jaffna", "Kilinochchi", "Mannar",
    "Vavuniya", "Mullaitivu", "Batticaloa", "Ampara", "Trincomalee",
    "Kurunegala", "Puttalam", "Anuradhapura", "Polonnaruwa", "Badulla",
    "Moneragala", "Ratnapura", "Kegalle",
]

_STREET_TYPES = ["Mawatha", "Road", "Lane", "Avenue", "Street", "Place"]

_TOWN_NAMES = [
    "Nugegoda", "Dehiwala", "Moratuwa", "Kelaniya", "Kaduwela", "Maharagama",
    "Pannipitiya", "Piliyandala", "Ragama", "Ja-Ela", "Seeduwa", "Negombo",
    "Kotte", "Battaramulla", "Malabe", "Homagama", "Padukka", "Kadawatha",
    "Wattala", "Hendala", "Minuwangoda", "Kandana", "Nittambuwa", "Veyangoda",
]


# ── Fake data generator ───────────────────────────────────────────────────────

class FakeSriLankan:
    """
    Deterministic fake data generator for Sri Lankan identity documents.

    Pass `seed` for reproducible datasets.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def full_name(self, ethnicity: str = "sinhala") -> str:
        if ethnicity == "tamil":
            given = self._rng.choice(_TAMIL_GIVEN_NAMES)
            surname = self._rng.choice(_TAMIL_SURNAMES)
        else:
            given = self._rng.choice(_SINHALA_GIVEN_NAMES)
            surname = self._rng.choice(_SINHALA_SURNAMES)
        # Some SL names use initials: "K. A. M. Perera"
        if self._rng.random() < 0.3:
            initials = " ".join(
                f"{self._rng.choice(string.ascii_uppercase)}."
                for _ in range(self._rng.randint(1, 3))
            )
            return f"{initials} {surname}"
        return f"{given} {surname}"

    def dob(
        self,
        min_age: int = 18,
        max_age: int = 70,
    ) -> date:
        today = date.today()
        age_days = self._rng.randint(min_age * 365, max_age * 365)
        return today - timedelta(days=age_days)

    def sex(self) -> str:
        return self._rng.choice(["M", "F"])

    # ── NIC ──────────────────────────────────────────────────────────────────

    def nic_number(self, dob: date | None = None, sex: str | None = None) -> str:
        """Generate a valid-format Sri Lankan NIC number (new 12-digit format)."""
        if dob is None:
            dob = self.dob()
        if sex is None:
            sex = self.sex()

        # Day of year (1-366); females add 500
        day_of_year = dob.timetuple().tm_yday
        if sex == "F":
            day_of_year += 500

        year_str = str(dob.year)
        doy_str = str(day_of_year).zfill(3)
        serial = str(self._rng.randint(0, 9999)).zfill(4)
        check = str(self._rng.randint(0, 9))
        return f"{year_str}{doy_str}{serial}{check}"

    def old_nic_number(self, dob: date | None = None, sex: str | None = None) -> str:
        """Generate an old-format NIC (9 digits + V/X)."""
        if dob is None:
            dob = self.dob(min_age=35, max_age=70)  # old NICs are for older people
        if sex is None:
            sex = self.sex()

        year_2digit = str(dob.year % 100).zfill(2)
        day_of_year = dob.timetuple().tm_yday
        if sex == "F":
            day_of_year += 500
        doy_str = str(day_of_year).zfill(3)
        serial = str(self._rng.randint(0, 9999)).zfill(4)
        suffix = self._rng.choice(["V", "X"])
        return f"{year_2digit}{doy_str}{serial}{suffix}"

    # ── Passport ─────────────────────────────────────────────────────────────

    def passport_number(self) -> str:
        """Generate a Sri Lankan passport number (N + 7 digits)."""
        letter = self._rng.choice("NPAEBCD")
        digits = "".join(self._rng.choices(string.digits, k=7))
        return f"{letter}{digits}"

    def passport_expiry(self, issued: date | None = None) -> date:
        """Passports valid for 5 or 10 years."""
        if issued is None:
            issued = date.today() - timedelta(days=self._rng.randint(0, 365 * 3))
        validity_years = self._rng.choice([5, 10])
        return issued.replace(year=issued.year + validity_years)

    def mrz_line1(self, surname: str, given_names: str, country: str = "LKA") -> str:
        """Generate TD3 MRZ line 1: P<COUNTRY + name field (44 chars)."""
        # Normalise: uppercase, replace spaces/non-alpha with <
        def mrz_name(s: str) -> str:
            s = s.upper().replace(" ", "<").replace("-", "<")
            import re
            return re.sub(r"[^A-Z<]", "<", s)

        name_field = f"{mrz_name(surname)}<<{mrz_name(given_names)}"
        return f"P<{country}{name_field:<{39}}"[:44]

    def mrz_line2(
        self,
        passport_no: str,
        nationality: str,
        dob: date,
        sex: str,
        expiry: date,
    ) -> str:
        """Generate TD3 MRZ line 2 (simplified, no check digits)."""
        dob_str = dob.strftime("%y%m%d")
        exp_str = expiry.strftime("%y%m%d")
        sex_ch = sex[0].upper()
        doc_no = f"{passport_no:<9}"[:9]
        personal = "<" * 14
        return f"{doc_no}0{nationality}{dob_str}{sex_ch}{exp_str}0{personal}0"[:44]

    # ── Driving License ───────────────────────────────────────────────────────

    def driving_license_number(self) -> str:
        """Generate a Sri Lankan driving license number (10-12 digits)."""
        return "".join(self._rng.choices(string.digits, k=self._rng.randint(10, 12)))

    def vehicle_categories(self) -> list[str]:
        all_cats = ["A", "A1", "B", "B1", "C", "D"]
        n = self._rng.randint(1, 4)
        return sorted(self._rng.sample(all_cats, min(n, len(all_cats))))

    def issue_date(self, dob: date | None = None) -> date:
        """Issue date: at least 18 years after DOB, not in the future."""
        if dob is None:
            dob = self.dob()
        earliest = dob.replace(year=dob.year + 18)
        latest = date.today()
        if earliest >= latest:
            return latest
        delta = (latest - earliest).days
        return earliest + timedelta(days=self._rng.randint(0, delta))

    def dl_expiry(self, issued: date) -> date:
        """DL valid for 5 years in Sri Lanka."""
        return issued.replace(year=issued.year + 5)

    # ── Address ──────────────────────────────────────────────────────────────

    def address(self) -> str:
        number = self._rng.randint(1, 999)
        street = f"{self._rng.choice(_TOWN_NAMES)} {self._rng.choice(_STREET_TYPES)}"
        town = self._rng.choice(_TOWN_NAMES)
        district = self._rng.choice(_DISTRICTS)
        return f"{number}, {street}, {town}, {district}"
