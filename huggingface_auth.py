import asyncio
import base64
import os
from datetime import datetime, timedelta
from getpass import getpass

import requests
from huggingface_hub import HfApi

from hivemind.proto.auth_pb2 import AccessToken
from hivemind.utils.auth import TokenAuthorizerBase
from hivemind.utils.crypto import RSAPublicKey
from hivemind.utils.logging import get_logger


logger = get_logger(__name__)


class HuggingFaceAuthorizer(TokenAuthorizerBase):
    _AUTH_SERVER_URL = 'https://collaborative-training-auth.huggingface.co'

    def __init__(self, experiment_id: int, username: str, password: str):
        super().__init__()

        self._experiment_id = experiment_id
        self._username = username
        self._password = password

        self._authority_public_key = None
        self._hf_api = HfApi()

    async def get_token(self) -> AccessToken:
        token = self._hf_api.login(self._username, self._password)

        try:
            url = f'{self._AUTH_SERVER_URL}/api/experiments/join/{self._experiment_id}/'
            headers = {'Authorization': f'Bearer {token}'}
            response = requests.put(url, headers=headers, json={
                'experiment_join_input': {
                    'peer_public_key': self.local_public_key.to_bytes().decode(),
                },
            })

            response.raise_for_status()
            response = response.json()

            self._authority_public_key = RSAPublicKey.from_bytes(response['auth_server_public_key'].encode())

            token_dict = response['hivemind_access']
            access_token = AccessToken()
            access_token.username = token_dict['username']
            access_token.public_key = token_dict['peer_public_key'].encode()
            access_token.expiration_time = str(datetime.fromisoformat(token_dict['expiration_time']))
            access_token.signature = token_dict['signature'].encode()

            logger.info(f'Access for user {access_token.username} '
                        f'has been granted until {access_token.expiration_time} UTC')
            return access_token
        finally:
            self._hf_api.logout(token)

    def add_collaborator(self) -> None:
        # This is a temporary workaround necessary until the experiment invite tokens are implemented.
        # It is not intended to be secure and designed to test the authorization code
        # without complicating the new user's joining procedure.

        token = self._hf_api.login('robot-bengali', base64.b64decode(b'aGdKQlViTDMzd2h2').decode())

        try:
            url = f'{self._AUTH_SERVER_URL}/api/experiments/{self._experiment_id}/'
            headers = {'Authorization': f'Bearer {token}'}
            response = requests.put(url, headers=headers, json={
                'experiment_full_update': {
                    'added_collaborators': [{'username': self._username}],
                },
            })

            response.raise_for_status()
            logger.info(f'User {self._username} has been added to collaborators of '
                        f'the experiment {self._experiment_id}')
        finally:
            self._hf_api.logout(token)

    def is_token_valid(self, access_token: AccessToken) -> bool:
        data = self._token_to_bytes(access_token)
        if not self._authority_public_key.verify(data, access_token.signature):
            logger.exception('Access token has invalid signature')
            return False

        try:
            expiration_time = datetime.fromisoformat(access_token.expiration_time)
        except ValueError:
            logger.exception(
                f'datetime.fromisoformat() failed to parse expiration time: {access_token.expiration_time}')
            return False
        if expiration_time.tzinfo is not None:
            logger.exception(f'Expected to have no timezone for expiration time: {access_token.expiration_time}')
            return False
        if expiration_time < datetime.utcnow():
            logger.exception('Access token has expired')
            return False

        return True

    _MAX_LATENCY = timedelta(minutes=1)

    def does_token_need_refreshing(self, access_token: AccessToken) -> bool:
        expiration_time = datetime.fromisoformat(access_token.expiration_time)
        return expiration_time < datetime.utcnow() + self._MAX_LATENCY

    @staticmethod
    def _token_to_bytes(access_token: AccessToken) -> bytes:
        return f'{access_token.username} {access_token.public_key} {access_token.expiration_time}'.encode()


DEFAULT_EXPERIMENT_ID = 3


def authorize_with_huggingface() -> HuggingFaceAuthorizer:
    experiment_id = os.getenv('HF_EXPERIMENT_ID', DEFAULT_EXPERIMENT_ID)

    username = os.getenv('HF_USERNAME')
    if username is None:
        username = input('HuggingFace username: ')
    password = os.getenv('HF_PASSWORD')
    if password is None:
        password = getpass('HuggingFace password: ')

    authorizer = HuggingFaceAuthorizer(experiment_id, username, password)
    authorizer.add_collaborator()
    asyncio.run(authorizer.refresh_token_if_needed())
    return authorizer
